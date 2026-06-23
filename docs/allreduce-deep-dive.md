# AllReduce 全链路深度解析

从 `dist.all_reduce()` 到 `ncclDevKernel_AllReduce_Sum_f32_RING_LL` 的完整调用链。

> **机器消费 & 实测验证（2026-06-21 更新）**
> - 本文档现在是**机器消费**的：`nccl_ptx_lib/chain/runtime_knowledge.py` 的 AllReduce `RuntimeRule`
>   把这里的 9 层编码进调用链产物（把文档 Layer 2 / Layer 5 各拆成两层）。
> - 已对照双卡 Qwen3-8B trace 实测验证（`nccl_ptx/call_chains.json`）：
>   两条匹配上的 NCCL 链路都复现了 9 层顺序——
>   ① f16 RING_LL（MLP `dist.all_reduce`，`shape=[1,16,4096]`，`grid=[16,1,1]`，`block=[512,1,1]`，`smem=88416`）；
>   ② f32 RING_LL（`dist.barrier`，`count=1`，`grid=[1,1,1]`，`block=[96,1,1]`，`smem=88416`）。
> - 实测**证实**：「双卡选 RING+LL」「9 层顺序」「mangled 符号」「18 条 ld/st.volatile.global」。
> - 实测**推翻**并已下文更正：`block=640`、`smem=0(LL)`、「<几KB→TREE+LL」。
> - 注意：第 9 层设备派发按 algo 后缀特化（`_RING→runRing`、`_TREE→runTree`）；
>   本地构建只特化了 **LL** collective kernel（LL128/SIMPLE 走通用 `ncclDevKernel_Generic`）。

---

## 给 NCCL 小白的速读版

如果你刚开始看 NCCL，可以先把这篇文档理解成一句话：

> PyTorch 只负责把 `tensor/count/dtype/op/stream` 交给 NCCL；NCCL 的 host 代码选择算法并
> 准备 `ncclWork`；真正跨 GPU 搬数据和做加法的是最后 launch 出来的 device kernel。

最小心智模型是三段：

| 新手问题 | 答案 | 本文对应 |
|---|---|---|
| 谁发起通信？ | Python 的 `dist.all_reduce()`，落到 PyTorch C++ 的 `ProcessGroupNCCL`。 | Layer 1-3 |
| 谁决定怎么通信？ | NCCL host 侧：检查参数、选 `RING+LL`、算 channel/线程、生成 `ncclWork`。 | Layer 4-7 |
| 谁真的搬数据？ | GPU 上的 `ncclDevKernel_AllReduce_...`，通过 peer GPU 的 ring buffer 做 `send/recv/reduce`。 | Layer 8-9 |

几个先记住的词：

| 词 | 可以先这么理解 |
|---|---|
| `rank` | 第几张 GPU。双卡时就是 rank 0 / rank 1。 |
| `comm` / communicator | 一组 GPU 的通信上下文，里面有 rank 数、channel、拓扑等。 |
| `channel` | NCCL 把大 tensor 切成多条“通信车道”并行跑；`grid.x` 通常就是活跃 channel 数。 |
| `algo` | 通信拓扑算法，例如 `RING` / `TREE` / `NVLS`。本次实测选中 `RING`。 |
| `proto` | 单次传输的协议，例如 `LL` / `LL128` / `SIMPLE`。本次实测选中 `LL`。 |
| `ncclWork` | host 侧写给 device kernel 的任务单：buffer 地址、count、chunk、funcIndex 都在里面。 |
| `channelMask` | 哪些 channel 参与本次 kernel 的 64-bit 位图；`grid.x = popcount(channelMask)`。 |
| `P2P` | 一张 GPU 能直接读写另一张 GPU 暴露出来的 memory，这是 Ring LL 的通信基础。 |

读这篇文档时建议按这个顺序：

1. 先看 [全链路总览](#1-全链路总览)，知道 9 层名字和每层交出的“产物”。
2. 再看 [算法选择策略](#74-算法选择策略)，理解为什么实测是 `RING+LL`。
3. 然后看 [ncclLaunchKernel](#8-第七层nccllaunchkernel--真正的-kernel-launch)，抓住 kernel 的 3 个参数、`grid/block/smem`。
4. 最后看 [RunWork](#10-第九层runwork--allreduce-设备端实现) 和 [PTX 转译要点](#12-ptx-转译要点)，这是迁移/转译最关键的部分。

如果你的目标是做 sunrise/pccl 兼容，先别被所有内部函数吓到。落地时最关键的是四件事：

- 对 PyTorch 暴露兼容的 NCCL 公共 API：至少 `ncclAllReduce`、communicator 初始化、类型/归约枚举。
- host 侧最终要构造出 device kernel 看得懂的 `ncclDevComm`、`ncclWork`、`channelMask`。
- runtime 必须能 launch 对应 PTX，并支持动态 shared memory、stream/event、跨卡 P2P 映射。
- device/ISA 转译必须覆盖 `ld/st.volatile.global`、`bar.sync`、`__popcll` 这些关键指令/语义。

---

## 目录

0. [给 NCCL 小白的速读版](#给-nccl-小白的速读版)
1. [全链路总览](#1-全链路总览)
2. [第一层：PyTorch Python API](#2-第一层pytorch-python-api)
3. [第二层：ProcessGroupNCCL::allreduce()](#3-第二层processgroupncclallreduce)
4. [第三层：collective() 模板 — NCCL 调用引擎](#4-第三层collective-模板--nccl-调用引擎)
5. [第四层：ncclAllReduce() — NCCL C API](#5-第四层ncclallreduce--nccl-c-api)
6. [第五层：Enqueue — 任务入队](#6-第五层enqueue--任务入队)
7. [第六层：Launch Prepare — 算法选择与工作构造](#7-第六层launch-prepare--算法选择与工作构造)
8. [第七层：ncclLaunchKernel — 真正的 kernel launch](#8-第七层nccllaunchkernel--真正的-kernel-launch)
9. [第八层：Device-Side — ncclKernelMain](#9-第八层device-side--ncclkernelmain)
10. [第九层：RunWork — AllReduce 设备端实现](#10-第九层runwork--allreduce-设备端实现)
11. [关键数据结构详解](#11-关键数据结构详解)
12. [PTX 转译要点](#12-ptx-转译要点)

---

## 1. 全链路总览

```
Python:  dist.all_reduce(tensor, op=ReduceOp.SUM)
  │
  ▼
Layer 1: torch.distributed.all_reduce()                         [Python]
  │
  ▼
Layer 2: ProcessGroupNCCL::allreduce(tensors, opts)             [C++]
  │  → allreduce_impl(tensor, opts)
  │  → collective(input, output, fn, OpType::ALLREDUCE)
  │
  ▼
Layer 3: fn(input, output, comm, stream)                        [lambda]
  │  → ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
  │
  ▼
Layer 4: ncclAllReduce()                                        [NCCL C API]
  │  → 构造 ncclInfo { ncclFuncAllReduce, ... }
  │  → ncclEnqueueCheck(&info)
  │
  ▼
Layer 5: taskAppend(comm, info)                                 [入队]
  │  → hostToDevRedOp() — ncclSum → ncclDevSum
  │  → collQueue.enqueue(info)
  │
  ▼
Layer 6: ncclLaunchPrepare → scheduleCollTasksToPlan            [调度]
  │  → topoGetAlgoInfo() — 选择 RING/TREE/NVLS
  │  → ncclDevFuncId() — 映射到 kernel 函数指针
  │  → initCollWorkElem() — 填充 ncclWorkElem
  │  → uploadWork() — 上传到 workFifo
  │
  ▼
Layer 7: ncclLaunchKernel(comm, plan)                           [launch]
  │  → grid = {channelCount, 1, 1}          // 数据量相关 (实测 f16=16, barrier=1)
  │  → block = {threadPerBlock, 1, 1}       // LL 上限 512, 调谐到 ≥96 (实测 f16=512, barrier=96)
  │  → smem = ncclShmemDynamicSize()        // 协议无关, sm_80 ≈ 88416 (实测)
  │  → args = {&devComm, &channelMask, &workHead}
  │  → cudaLaunchKernelExC(...) / cudaLaunchKernel(...)
  │
  ▼
Layer 8: ncclDevKernel_AllReduce_Sum_f32_RING_LL                [__global__]
  │  → ncclKernelMain<SpecializedFnId, RunWork<...>>()
  │  → 加载 ncclDevComm/Channel/Work 到 shared memory
  │
  ▼
Layer 9: RunWork<AllReduce, f32, Sum, RING, LL>::run()          [device]
  │  → runRing<f32, Sum, ProtoLL>(args)      // _RING → runRing; _TREE → runTree
  │  → Primitives: send → recvReduceSend → ... → recv
```

### barrier 变体（同一条 NCCL 机器，不同入口）

NCCL **没有** `ncclBarrier` / `ncclFuncBarrier` collective（`grep src/` 无此符号）。
`torch.distributed.barrier()` 复用 AllReduce 机器：在缓存的 1 元素 `barrierTensor_`
上跑一次 `ncclAllReduce(ncclFuncAllReduce)`。所以 **Layer 4–9 与 `dist.all_reduce()`
完全相同**，只有入口两层不同：

```
Layer 1: torch.distributed.barrier()                             [Python]   (c10d/distributed_c10d.py:4122)
Layer 2: ProcessGroupNCCL::barrier()                             [C++]      (header:610, 用 barrierTensor_ @358)
   └→ allreduce_impl(barrierTensor_, opts) → collective() → ncclAllReduce(...)   ← 接回 Layer 4
```

实测：`dist.barrier()` 触发的 `ncclDevKernel_AllReduce_Sum_f32_RING_LL`（`count=1`，
`corr=28537`）就是这条路径。工具默认把这类 barrier 同步 kernel 排除在 used-only 集合
外（见 `docs/adr/003-call-chain-tracing.md` 的 `--include-sync-kernels`）。

---

## 2. 第一层：PyTorch Python API

```python
# 用户代码
import torch.distributed as dist
tensor = torch.randn(4096, device="cuda")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

**调用路径：**
```
torch.distributed.all_reduce()
  → tensor.new_empty() 创建 output
  → default_pg.allreduce([tensor], AllreduceOptions(reduceOp=SUM))
  → 返回 Work 对象 (异步句柄)
```

**Python 层做的事情：**
- 校验输入 tensor（GPU 上、连续、非空）
- 获取当前 process group
- 调用 C++ 层的 `allreduce()`
- 返回 `Work` 异步句柄

---

## 3. 第二层：ProcessGroupNCCL::allreduce()

**源码位置：** `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`

### 3.1 入口函数

```cpp
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // 1. 校验: 只支持单 tensor
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();

  // 2. 复数 tensor → view as real
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }

  // 3. 节点内快速路径 (自定义 SHMEM allreduce, 绕过 NCCL)
  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
    if (algo != AllReduceAlgo::NONE) {
      intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>();
    }
  }

  // 4. 进入 NCCL 路径
  return allreduce_impl(tensor, opts);
}
```

### 3.2 allreduce_impl — 调用 NCCL

```cpp
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,    // ← in-place: input == output
      [&](at::Tensor& input, at::Tensor& output,
          ncclComm_t comm, at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),     // sendbuff
            output.data_ptr(),    // recvbuff (同 sendbuff = in-place)
            input.numel(),        // count (元素个数)
            ncclDataType,         // ncclDataType_t
            ncclReduceOp,         // ncclRedOp_t
            comm,                 // ncclComm_t
            stream.stream());     // cudaStream_t
      },
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}
```

### 3.3 类型映射

**数据类型映射 (getNcclDataType):**

| PyTorch `ScalarType` | NCCL `ncclDataType_t` | 字节数 |
|---|---|---|
| `kFloat` (float32) | `ncclFloat32` | 4 |
| `kHalf` (float16) | `ncclFloat16` | 2 |
| `kBFloat16` | `ncclBfloat16` | 2 |
| `kDouble` (float64) | `ncclFloat64` | 8 |
| `kInt` (int32) | `ncclInt32` | 4 |
| `kLong` (int64) | `ncclInt64` | 8 |
| `kChar` (int8) | `ncclInt8` | 1 |
| `kByte` (uint8) | `ncclUint8` | 1 |
| `kBool` | `ncclUint8` | 1 |

**归约操作映射 (getNcclReduceOp):**

| PyTorch `ReduceOp` | NCCL `ncclRedOp_t` | 说明 |
|---|---|---|
| `SUM` | `ncclSum` | 求和 |
| `AVG` | `ncclAvg` | 平均 (NCCL 2.10+) |
| `PRODUCT` | `ncclProd` | 乘积 |
| `MIN` | `ncclMin` | 最小值 |
| `MAX` | `ncclMax` | 最大值 |
| `PREMUL_SUM` | 动态 op via `ncclRedOpCreatePreMulSum` | 预乘求和 |

**特殊处理：** `bool` + `SUM` → 映射为 `ncclMax`（避免 uint8 溢出）。

---

## 4. 第三层：collective() 模板 — NCCL 调用引擎

```cpp
template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,                    // ← ncclAllReduce 的 lambda
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle) {

  // ① 获取/创建缓存的 NCCL communicator
  auto ncclComm = getNCCLComm(key, device, opType);

  // ② 获取专用 NCCL stream，与计算 stream 同步
  auto ncclStream = ncclStreams_.at(key);
  syncStream(device, ncclEvents_[key], ncclStream);

  // ③ 创建 Work 追踪对象
  auto work = initWork(device, rank_, opType, profilingTitle, inputs, outputs);

  // ④ 为 CUDA caching allocator 记录 stream 使用
  for (auto& input : inputs)
    c10::cuda::CUDACachingAllocator::recordStream(
        input.storage().data_ptr(), ncclStream);

  // ⑤ ★ 调用 ncclAllReduce ★
  C10D_NCCL_CHECK(fn(inputs[0], outputs[0], comm, ncclStream));

  // ⑥ 记录结束事件，交给 watchdog 线程
  work->ncclEndEvent_->record(ncclStream);
  workEnqueue(work);

  return work;
}
```

**关键设计：**
- **communicator 缓存**：同一组 device+opType 复用 `ncclComm_t`，避免重复初始化
- **独立 NCCL stream**：通信与计算异步重叠
- **Work 对象**：异步句柄，支持 `.wait()` / `.synchronize()`

---

## 5. 第四层：ncclAllReduce() — NCCL C API

**源码位置：** `nccl/src/collectives.cc:30-52`

```c
ncclResult_t ncclAllReduce(
    const void* sendbuff,      // 输入数据指针 (device memory)
    void* recvbuff,            // 输出数据指针 (device memory, 可与 sendbuff 相同 = in-place)
    size_t count,              // 元素个数 (不是字节数!)
    ncclDataType_t datatype,   // 数据类型 (ncclFloat32, ncclFloat16, ...)
    ncclRedOp_t op,            // 归约操作 (ncclSum, ncclProd, ...)
    ncclComm* comm,            // 通信器 (包含 rank, nRanks, channels 等)
    cudaStream_t stream        // CUDA stream (异步执行)
) {
  // 构造 ncclInfo 描述结构
  struct ncclInfo info = {
    ncclFuncAllReduce,     // 操作类型枚举 (= 4)
    "AllReduce",           // 操作名称 (用于 profiling)
    sendbuff, recvbuff,    // 数据指针
    count, datatype, op,   // 操作参数
    0,                     // root (AllReduce 不需要)
    comm, stream,          // 通信器和 stream
    ALLREDUCE_CHUNKSTEPS,  // = NCCL_STEPS/2 = 4
    ALLREDUCE_SLICESTEPS   // = NCCL_STEPS/4 = 2
  };

  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}
```

**参数详解：**

| 参数 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `sendbuff` | `const void*` | 输入数据 (device memory) | `tensor.data_ptr()` |
| `recvbuff` | `void*` | 输出数据 (可与 sendbuff 相同) | 同上 (in-place) |
| `count` | `size_t` | **元素个数** (非字节) | `4096` (4096个f32 = 16KB) |
| `datatype` | `ncclDataType_t` | 元素类型 | `ncclFloat32` |
| `op` | `ncclRedOp_t` | 归约操作 | `ncclSum` |
| `comm` | `ncclComm*` | 通信器上下文 | 初始化时创建 |
| `stream` | `cudaStream_t` | CUDA 异步流 | PyTorch 的 NCCL stream |

---

## 6. 第五层：Enqueue — 任务入队

**源码位置：** `nccl/src/enqueue.cc`

### 6.1 ncclEnqueueCheck

```c
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  ncclGroupStartInternal();           // 开始 group 语义

  CommCheck(info->comm, ...);         // 校验 communicator
  ncclCommEnsureReady(info->comm);    // 确保初始化完成
  ArgsCheck(info);                    // 校验参数

  taskAppend(info->comm, info);       // ★ 入队 ★

  ncclGroupEndInternal();             // 可能触发实际 launch (depth==1 时)
}
```

### 6.2 taskAppend — 核心入队逻辑

```c
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  // ① 归约操作转换: host → device
  hostToDevRedOp(&info->opFull, info->op, info->datatype, comm);
  //   ncclSum  → ncclDevSum
  //   ncclAvg  → ncclDevPreMulSum (float, 附带 1/nRanks 标量)
  //   ncclAvg  → ncclDevSumPostDiv (int, 后续除以 nRanks)

  // ② 单 rank 快速路径: 直接 memcpy, 不 launch kernel
  if (comm->nRanks == 1) {
    ncclLaunchOneRank(info->recvbuff, info->sendbuff,
                      info->count, info->opFull, info->datatype, info->stream);
    return ncclSuccess;
  }

  // ③ 多 rank: 加入 group, 入队到 collQueue
  ncclGroupCommJoin(info->comm);

  struct ncclInfo* t = ncclMemoryStackAlloc<struct ncclInfo>(&comm->memScoped);
  info->nChannels = 0;
  info->nThreads = 0;
  info->algorithm = NCCL_ALGO_UNDEF;   // 算法待选择
  info->protocol = NCCL_PROTO_UNDEF;   // 协议待选择
  info->userTuned = false;             // 未用用户 tuner (enqueue.cc:1972)
  memcpy(t, info, sizeof(struct ncclInfo));
  ncclIntruQueueSortEnqueue(&tasks->collQueue, t, collCmp);
  // (随后还有 stream 追踪尾巴 :1980-2003, 此处省略)
}
```

### 6.3 hostToDevRedOp — 归约操作转换

| 用户操作 | 设备操作 | 说明 |
|----------|---------|------|
| `ncclSum` | `ncclDevSum` | 直接求和 |
| `ncclProd` | `ncclDevProd` | 直接求积 |
| `ncclMin` | `ncclDevMinMax` | XOR 符号位后求 min (处理有符号数) |
| `ncclMax` | `ncclDevMinMax` | XOR 全部位后求 min |
| `ncclAvg` (float) | `ncclDevPreMulSum` | 每个 rank 先乘 `1/nRanks`，再求和 |
| `ncclAvg` (int) | `ncclDevSumPostDiv` | 先求和，最后除以 `nRanks` |

---

## 7. 第六层：Launch Prepare — 算法选择与工作构造

### 7.1 scheduleCollTasksToPlan

**源码位置：** `nccl/src/enqueue.cc:743-870`

```c
ncclResult_t scheduleCollTasksToPlan(struct ncclComm* comm,
                                      struct ncclKernelPlan* plan,
                                      int* nWorkBudget) {
  // 对 collQueue 中的每个 collective:

  // ① 算法+协议选择 (tuner)
  getTunerInfo(collInfo);              // 插件 tuner
  // 或
  topoGetAlgoInfo(collInfo);           // 暴力搜索所有 algo×proto 组合

  // ② channel 数和线程数
  getChannnelThreadInfo(collInfo);     // 基于数据量和拓扑

  // ③ 计算 kernel 函数索引
  collInfo->workFuncIndex = ncclDevFuncId(
      collInfo->coll,       // ncclFuncAllReduce
      collInfo->opFull.op,  // ncclDevSum
      collInfo->datatype,   // ncclFloat32
      collInfo->algorithm,  // NCCL_ALGO_RING
      collInfo->protocol    // NCCL_PROTO_LL
  );

  // ④ 获取数据分布 pattern
  getPatternInfo(collInfo);
  //   AllReduce + RING → ncclPatternRingTwice
  //   AllReduce + TREE → ncclPatternTreeUpDown

  // ⑤ 计算 chunk 参数
  computeCollChunkInfo(collInfo);

  // ⑥ 填充 ncclWorkElem
  initCollWorkElem(collInfo, &work);

  // ⑦ 映射到 kernel 函数指针
  plan->kernelFn = ncclDevKernelForFunc[collInfo->workFuncIndex];
  plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[collInfo->workFuncIndex];
}
```

> 注：上面 ⑤⑥⑦ 其实跑在各 `add*ToPlan` 辅助函数里（`addCBDCollToPlan`
> `enqueue.cc:844`、`addCollnetCollToPlan :855`、`addTunedCollToPlan :865`），
> `scheduleCollTasksToPlan` 按拓扑派发到其中之一。`computeCollChunkInfo` 在
> `:276/:348/:450`，`initCollWorkElem` 在 `:277/:349/:452`。引用的 `743-870` 行段准确，
> 此处为忠实简化。

### 7.2 ncclDevFuncId — 函数索引映射

```c
// device.h:443-498
inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  if (coll == ncclFuncAllReduce) {
    // 线性索引 = ((devRedOp * NumTypes + type) * nAlgos + algo) * NumProtocols + proto
    // nAlgos = 6 (RING, TREE, COLLNET_DIRECT, COLLNET_CHAIN, NVLS, NVLS_TREE)
    // NumProtocols = 3 (LL, LL128, SIMPLE)
    // NumTypes = 10 (int8, uint8, int32, uint32, int64, uint64, float16, float32, float64, bfloat16)
    row += ((devRedOp * NumTypes + type) * nAlgos + algo) * NCCL_NUM_PROTOCOLS + proto;
  }
  return ncclDevFuncRowToId[row];  // 查表得到实际 funcIndex
}
```

**示例：** AllReduce + Sum + f32 + RING + LL
```
devRedOp = ncclDevSum   = 0   // device.h:27
type     = ncclFloat32  = 7   // nccl.h.in:213  (注意 ncclFloat16=6 @:211)
algo     = NCCL_ALGO_RING = 1 // nccl_common.h:31  (NCCL_ALGO_TREE=0 @:30)
proto    = NCCL_PROTO_LL  = 0 // nccl_common.h:39

row = ((devRedOp * 10 + type) * 6 + algo) * 3 + proto
    = ((0   * 10 + 7)    * 6 + 1)   * 3 + 0
    = (7 * 6 + 1) * 3 = 43 * 3 = 129
funcIndex = ncclDevFuncRowToId[129]
```

> 注：早先版本误把 `ncclFloat32` 写成 6、`NCCL_ALGO_RING` 写成 0，算出 `row=108`，
> 已据 `nccl.h.in` / `nccl_common.h` 实际枚举更正为 `row=129`。

### 7.3 initCollWorkElem — 工作元素构造

```c
static ncclResult_t initCollWorkElem(struct ncclInfo* collInfo,
                                      struct ncclWorkElem* work) {
  work->sendbuff      = collInfo->sendbuff;    // 输入数据指针
  work->recvbuff      = collInfo->recvbuff;    // 输出数据指针
  work->root          = collInfo->root;        // root rank (AllReduce 不用)
  work->count         = collInfo->count;       // 总元素数
  work->nWarps        = collInfo->nThreads / WARP_SIZE;  // warp 数
  work->redOpArg      = collInfo->opFull.scalarArg;      // 归约参数 (如 1/nRanks)
  work->redOpArgIsPtr = collInfo->opFull.scalarArgIsPtr;
  work->chunkCount    = collInfo->chunkCount;  // 每个 chunk 的元素数
  work->isUsed        = 1;
  work->oneNode       = (collInfo->comm->nNodes == 1);
}
```

### 7.4 算法选择策略

真实选择器是 `topoGetAlgoInfo`（`src/enqueue.cc:1484-1538`）：**暴力遍历所有
algo×proto 组合**，对每个组合调 `ncclTopoGetAlgoTime` 估时，取**最小时间**的那一对。
不是按数据量查表。

实测（双卡 Qwen3-8B，2 rank / 单节点 / NVLink）：

| 触发 | shape | count (元素) | 选中 algo×proto | 备注 |
|------|-------|------------|-----------------|------|
| MLP `dist.all_reduce` (f16) | `[1,16,4096]` | 65536 (131072 B) | **RING + LL** | `grid=[16,1,1]` |
| `dist.barrier` (f32) | `[1]` | 1 (4 B) | **RING + LL** | `grid=[1,1,1]` |

可见在 2 rank / NVLink 拓扑下，**连 4 字节的极小消息都选 RING+LL**——RING 的
每跳延迟在该拓扑下低于 TREE。下表是「经验性启发」，**不保证与真实选择器一致**，
仅作粗略参考：

| 经验条件 | 常见 algo | 常见 proto |
|---------|----------|----------|
| 小~中数据 | RING | LL / LL128 |
| 大数据 | RING / NVLS | SIMPLE |
| NVSwitch 可用 + 大数据 | NVLS | SIMPLE |
| CollNet 可用 | COLLNET_DIRECT | SIMPLE |

> 注：本地 `libnccl.so` **编译进**了 `AllReduce_Sum_{u8,f16,f32,f64,u32,u64,bf16}_{RING,TREE}_LL`
> 共 14 个特化 collective kernel，但 `topoGetAlgoInfo` 在 2 rank/NVLink 下**只选中
> RING+LL**——TREE kernel 编译了却没被选。`nccl_ptx/` 的 used-only 过滤因此只保留
> 实际 launch 的 2 个 RING_LL kernel。

---

## 8. 第七层：ncclLaunchKernel — 真正的 kernel launch

**源码位置：** `nccl/src/enqueue.cc:1351-1409`

```c
ncclResult_t ncclLaunchKernel(struct ncclComm* comm,
                               struct ncclKernelPlan* plan) {
  void *fn = plan->kernelFn;     // 从 ncclDevKernelForFunc[] 查到的函数指针
                                  // 例如: ncclDevKernel_AllReduce_Sum_f32_RING_LL

  cudaStream_t launchStream = tasks->streams->stream;

  dim3 grid  = {(unsigned)plan->channelCount, 1, 1};   // 每个 channel 一个 block (数据量相关)
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1}; // LL 上限 512, 经 getChannnelThreadInfo 调谐

  size_t smem = ncclShmemDynamicSize(comm->cudaArch);
  // ⚠ 协议无关: 取所有协议的 max (见 device.h:395-408)
  //   ncclShmemScratchWarpSize = max_constexpr{
  //     LL: 0,                                 ← LL 分支确实是 0
  //     LL128: (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE)*sizeof(uint64_t),
  //     SIMPLE:(ncclCollUnroll*WARP_SIZE+1)*16,
  //     NVLS:  ... }
  //   → max 取 SIMPLE 的非零值, 对**所有** kernel (含 LL 特化) 都返回非零
  //   smem = warpScratch * (NCCL_MAX_NTHREADS/WARP_SIZE)  // device.h:406-408
  //   sm_80 上 ≈ 88416 字节 (实测 f16 allreduce 与 f32 barrier 的 RING_LL 都是 88416)

  // ★ 三个 kernel 参数 ★
  void *args[3] = {
    &comm->devComm,         // ncclDevComm* — 通信上下文
    &plan->channelMask,     // uint64_t — 活跃 channel 位图
    &plan->workHead         // ncclWork* — 工作描述 FIFO 头
  };

  // CUDA 11.8+ 路径: 使用 launch attributes
  #if CUDART_VERSION >= 11080
  if (driverVersion >= 11080) {
    cudaLaunchConfig_t launchConfig = {0};
    launchConfig.gridDim = grid;
    launchConfig.blockDim = block;
    launchConfig.dynamicSmemBytes = smem;
    launchConfig.stream = launchStream;
    // attrs: cluster dimension (sm90), scheduling policy, mem sync domain
    cudaLaunchKernelExC(&launchConfig, fn, args);
    return ncclSuccess;
  }
  #endif

  // 标准 launch
  cudaLaunchKernel(fn, grid, block, args, smem, launchStream);
  return ncclSuccess;
}
```

### 参数构造详解

| 参数 | 值 | 构造方式 |
|------|-----|---------|
| `comm->devComm` | `ncclDevComm*` | `ncclCommInitRank()` 时构建，存在 device memory 上 |
| `plan->channelMask` | `uint64_t` | 每 bit 代表一个 channel 是否参与此次操作 |
| `plan->workHead` | `ncclWork*` | `uploadWork()` 写入 workFifo 后的头指针 |
| `grid` | `{channelCount, 1, 1}` | 数据量相关，`getChannnelThreadInfo`(`enqueue.cc:1574-1588`) 决定；实测 f16 allreduce=16, f32 barrier=1 |
| `block` | `{threadPerBlock, 1, 1}` | LL 上限 `NCCL_LL_MAX_NTHREADS=512`(`device.h:57`)，`getChannnelThreadInfo`(`enqueue.cc:1601-1614`) 按数据量减半，最低 3-warp=96；实测 f16=512, f32 barrier=96 |
| `smem` | ≈88416 (sm_80) | `ncclShmemDynamicSize()`，**协议无关**取 max（见上），含 LL 特化 kernel；不是 0 |

> 注：早先版本把 `block` 写成「`NCCL_MAX_NTHREADS` 固定 640」、`smem` 写成「0 (LL)」，
> 均与实测/源码不符，已更正。640 是 LL128/SIMPLE/NVLS 的上限；LL 用 512 并调谐。

### uploadWork — 工作上传

```c
static ncclResult_t uploadWork(struct ncclComm* comm,
                                struct ncclKernelPlan* plan) {
  // 非持久化: 使用 comm->workFifoHeap (host-pinned 或 GDR 内存)
  // 持久化:   ncclCudaMalloc + ncclCudaMemcpy 到 device memory

  for (int c = 0; c < channelUbound; c++) {
    // 将每个 channel 的 ncclWork 结构拷贝到 FIFO
    workHeap[ix & ixMask] = q->work;
  }

  // 设置 workHead 指针
  plan->workHead = &comm->devWorkFifoHeap[ixHead & ixMask];
}
```

---

## 9. 第八层：Device-Side — ncclKernelMain

**源码位置：** `nccl/src/device/common.h:124-213`

### 9.1 Kernel 入口 (由 DEFINE_ncclDevKernel 宏生成)

```c
// 宏展开后:
__global__ void ncclDevKernel_AllReduce_Sum_f32_RING_LL(
    struct ncclDevComm* comm,
    uint64_t channelMask,
    struct ncclWork* workHead) {
  ncclKernelMain<
      SpecializedFnId,   // 编译时确定的函数 ID
      RunWork<ncclFuncAllReduce, float, FuncSum<float>, NCCL_ALGO_RING, NCCL_PROTO_LL>
  >(comm, channelMask, workHead);
}
```

### 9.2 ncclKernelMain — 通用 kernel 主体

```c
template<int SpecializedFnId, typename SpecializedRunWork>
__device__ void ncclKernelMain(struct ncclDevComm* comm,
                                uint64_t channelMask,
                                struct ncclWork* workHead) {
  int tid = threadIdx.x;  // 0..639

  // ═══ 阶段 1: blockIdx → channelId 映射 ═══
  // channelMask 是一个 64-bit 位图, 每个 bit 代表一个 channel
  // blockIdx.x 是第几个活跃 channel (通过 popcount 计算)
  if (tid < WARP_SIZE) {
    int x = tid;
    if (channelMask & (1ull << x)) {
      int y = __popcll(channelMask & ((1ull << x) - 1));
      if (blockIdx.x == y) ncclShmem.channelId = x;
    }
  }
  __syncthreads();

  // ═══ 阶段 2: 用前 3 个 warp 并行加载数据到 shared memory ═══
  switch (tid / WARP_SIZE) {
  case 0:  // warp 0: 加载 ncclDevComm
    copyToShmem16(tid % WARP_SIZE, &ncclShmem.comm, comm, sizeof(ncclDevComm));
    break;
  case 1:  // warp 1: 加载 ncclDevChannel
    copyToShmem16(tid % WARP_SIZE, &ncclShmem.channel,
                  &((ncclDevCommAndChannels*)comm)->channels[channelId],
                  sizeof(ncclDevChannel));
    break;
  case 2:  // warp 2: 加载 ncclWork
    copyToShmem16(tid % WARP_SIZE, &ncclShmem.work,
                  workHead + blockIdx.x, sizeof(ncclWork));
    break;
  }
  __syncthreads();

  // ═══ 阶段 3: 主循环 — 处理 work 链表 ═══
  while (true) {
    // 如果是最后一个 work item, 通知 host 完成
    if (tid == 0 && ncclShmem.work.header.isLast && ncclShmem.work.header.inFifo) {
      *ncclShmem.channel.workFifoDone = ncclShmem.work.header.doneAcks;
    }

    // 解引用 redOpArg 指针 (如果是间接引用)
    if (ncclShmem.work.header.type == ncclWorkTypeColl) {
      if (tid < NCCL_MAX_WORK_ELEMENTS)
        ncclRedopPtrDeref(&ncclShmem.work.elems[tid]);
    }
    __syncthreads();

    // ═══ 阶段 4: 分发到具体的 RunWork 实现 ═══
    if (0 <= SpecializedFnId &&
        ncclShmem.work.header.funcIndex == (unsigned)SpecializedFnId) {
      // 快速路径: 编译时特化, 直接调用
      SpecializedRunWork().run(&ncclShmem.work);
    } else {
      // 慢速路径: 通过函数表间接调用 (Generic kernel)
      ncclDevFuncTable[ncclShmem.work.header.funcIndex]();
    }

    // ═══ 阶段 5: 获取下一个 work 或退出 ═══
    if (ncclShmem.work.header.isLast) break;
    copyToShmem16(tid, &ncclShmem.work, workHead + workIxNext, sizeof(ncclWork));

    // 检查 abort flag
    int aborted = tid == 0 ? *comm->abortFlag : 0;
    if (barrierReduceAny(aborted)) break;
  }
}
```

**PTX 中对应的关键指令：**

| 阶段 | PTX 指令 | 说明 |
|------|---------|------|
| 阶段 1 | `mov.u32 %r1, %tid.x` + `setp.gt.s32` | `tid < 32` 检查 |
| 阶段 2 | `ld.param.u64` + `cvta.shared` + 循环 `ld.global` | 参数加载 + 拷贝到 shared |
| 阶段 3 | `bar.sync 0` | 全 block 同步 |
| 阶段 4 | 直接内联 (特化路径无间接跳转) | 编译器展开 |
| 阶段 5 | `bar.sync 0` + `ld.volatile.global` | 同步 + 检查 abort |

---

## 10. 第九层：RunWork — AllReduce 设备端实现

**源码位置：** `nccl/src/device/all_reduce.h`

### 10.1 RunWork — 遍历 work 元素

```c
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;  // 当前 warp ID (0..19)
    ncclWorkElem* we = &w->elems[0];

    // 遍历 ncclWork 中的所有 workElem
    while (we->isUsed) {
      if (wid < we->nWarps) {
        // 当前 warp 负责这个 workElem
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(we);
      }
      we = nextElem(we);
    }
  }
};
```

### 10.2 RunWorkElement — AllReduce Ring LL 的实现

```c
// all_reduce.h:716-721
template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};
```

> **algo 派发**：`RunWorkElement` 按 `Algo` 模板参数特化——`NCCL_ALGO_RING → runRing`，
> `NCCL_ALGO_TREE → runTree`（`getPatternInfo` `enqueue.cc:1633` 把
> `AllReduce+TREE → ncclPatternTreeUpDown`）。algo 编码在 kernel 名后缀里
> （`_RING` / `_TREE`），所以 `nccl_ptx_lib/chain/runtime_knowledge.py` 据此特化末层：
> `_RING → RunWork<…>::run() → runRing<Proto>`、`_TREE → … → runTree`。

### 10.3 runRing — Ring AllReduce 核心算法 (LL 协议)

```c
template<typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runRing(ncclWorkElem *args) {
  const int tid = threadIdx.x;
  const int nthreads = (int)args->nWarps * WARP_SIZE;
  ncclRing *ring = &ncclShmem.channel.ring;
  int ringIx = ring->index;

  const int nranks = ncclShmem.comm.nRanks;    // 总 GPU 数
  ssize_t chunkCount = args->chunkCount;        // 每个 chunk 的元素数
  const ssize_t loopCount = nranks * chunkCount;
  ssize_t gridOffset = args->workOffset;
  ssize_t channelCount = args->workCount;

  // 构造通信原语对象
  // FanSymmetric<1>: 1 个 sender (prev), 1 个 receiver (next)
  Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
      tid, nthreads,
      &ring->prev, &ring->next,    // 通信对端 rank
      args->sendbuff,               // 输入 buffer
      args->recvbuff,               // 输出 buffer
      args->redOpArg);              // 归约参数

  // ═══ Ring AllReduce 算法 ═══
  // 总步骤数 = 2*(nranks-1)
  // 前 (nranks-1) 步: Reduce-Scatter (每个 GPU 得到一个完整 chunk)
  // 后 (nranks-1) 步: AllGather (广播结果)

  for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
    ssize_t offset = gridOffset + elemOffset;
    ssize_t nelem = min(chunkCount, channelCount - elemOffset);

    // ─── 第 1 步: 发送自己的 chunk 给 next ───
    int chunk = modRanks(ringIx + nranks - 1);
    prims.send(offset + chunk * chunkCount, nelem);

    // ─── 第 2 ~ (nranks-1) 步: 接收 → 归约 → 转发 ───
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      prims.recvReduceSend(offset + chunk * chunkCount, nelem);
    }

    // ─── 第 nranks 步: 接收最后一个 → 归约 → 存储 → 转发 ───
    chunk = ringIx;
    prims.directRecvReduceCopySend(
        offset + chunk * chunkCount,   // recv from prev
        offset + chunk * chunkCount,   // store to output
        nelem, /*postOp=*/true);

    // ─── AllGather 阶段: (nranks-2) 步转发 ───
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - 1 - j);
      prims.directRecvCopySend(offset + chunk * chunkCount, nelem);
    }

    // ─── 最后一步: 接收最终结果 (不需要归约) ───
    chunk = modRanks(ringIx + 1);
    prims.directRecv(offset + chunk * chunkCount, nelem);
  }
}
```

### 10.4 Ring AllReduce 图示 (3 GPU 示例)

```
GPU 0 ─────→ GPU 1 ─────→ GPU 2 ─────→ GPU 0   (Ring 拓扑)
  ↑                                              │
  └──────────────────────────────────────────────┘

初始:  GPU0=[A0,B0,C0]  GPU1=[A1,B1,C1]  GPU2=[A2,B2,C2]

── Reduce-Scatter 阶段 (2 步) ──

步 1: 每个 GPU 发送自己的最后一个 chunk
  GPU0 → C0 → GPU1
  GPU1 → C1 → GPU2
  GPU2 → C2 → GPU0

步 2: 接收 → 归约 → 转发
  GPU0: recv C2, reduce C0+C2, send to GPU1
  GPU1: recv C0, reduce C1+C0, send to GPU2
  GPU2: recv C1, reduce C2+C1, send to GPU0
  ...

步 nranks: 最终归约 + 存储
  GPU0: recv+reduce+store C (完整结果)
  ...

── AllGather 阶段 (2 步) ──

步 1: 转发完整 chunk
  GPU0 → C_complete → GPU1
  ...

步 2: 接收最终 chunk
  GPU0: recv B_complete (来自 GPU2)
  ...

最终:  GPU0=GPU1=GPU2=[A0+A1+A2, B0+B1+B2, C0+C1+C2]
```

### 10.5 ProtoLL — Low-Latency 协议实现

LL 协议的核心是用 **data + flag** 打包在 64-bit word 中：

```
┌──────────────────────────────────────┐
│  32-bit data  │  16-bit flag  │ 16b  │
└──────────────────────────────────────┘

发送: st.volatile.global.v4.u32 [addr], {data_lo, flag, data_hi, flag}
接收: ld.volatile.global.b32 flag, [flag_addr]
      while (flag != expected) { /* spin */ }
      ld.volatile.global.v4.u32 {data_lo, _, data_hi, _}, [addr]
```

**这就是 PTX 中那 18 条 `ld/st.volatile.global` 的用途** — GPU 间通过 global memory 上的 volatile 读写来同步数据。

---

## 11. 关键数据结构详解

### 11.1 ncclDevComm (通信上下文)

```c
struct ncclDevComm {
  int rank;                        // 当前 GPU 的 rank
  int nRanks;                      // 总 GPU 数
  int node;                        // 当前节点编号
  int nNodes;                      // 总节点数
  int buffSizes[NCCL_NUM_PROTOCOLS]; // 每种协议的 ring buffer 大小
  int p2pChunkSize;                // P2P chunk 大小

  int workFifoDepth;               // work FIFO 深度
  struct ncclWork* workFifoHeap;   // work FIFO (device memory)

  int* collNetDenseToUserRank;     // CollNet rank 映射
  volatile uint32_t* abortFlag;    // 中断标志 (host 可写)

  struct ncclDevChannel* channels; // channel 数组 (包含 ring/tree 拓扑)
};
```

### 11.2 ncclDevChannel (通信 channel)

```c
struct ncclDevChannel {
  struct ncclDevChannelPeer** peers;  // 每个 peer 的连接信息

  struct ncclRing ring;               // Ring 拓扑
    // .prev — ring 中上一个 rank
    // .next — ring 中下一个 rank
    // .userRanks — rank 映射表
    // .index — 当前 rank 在 ring 中的位置

  struct ncclTree tree;               // Tree 拓扑
    // .depth — 树深度
    // .up — 父节点 rank
    // .down[3] — 子节点 rank

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;
  struct ncclNvls nvls;

  uint32_t* workFifoDone;            // 完成计数器
};
```

### 11.3 ncclWork (工作描述, 512 字节)

```c
struct ncclWork {
  struct ncclWorkHeader header;
  // header.funcIndex — 对应哪个 kernel 函数
  // header.isLast — 是否是最后一个 work
  // header.inFifo — 是否在 FIFO 中
  // header.type — ncclWorkTypeColl / ncclWorkTypeRegColl
  // header.workNext — 下一个 work 的偏移

  union {
    struct ncclWorkElem elems[9];       // 最多 9 个 collective 元素
    struct ncclWorkElemP2p p2pElems[16];
    struct ncclWorkElemReg regElems[2];
  };
};
```

### 11.4 ncclWorkElem (工作元素)

```c
struct ncclWorkElem {
  uint8_t isUsed:1;         // 是否有效
  uint8_t redOpArgIsPtr:1;  // redOpArg 是指针还是标量
  uint8_t oneNode:1;        // 是否单节点
  uint8_t regUsed;          // 是否使用注册 buffer
  uint8_t nWarps;           // 分配给此操作的 warp 数
  uint8_t direct;           // direct 标志

  uint32_t root;            // root rank (Broadcast/Reduce 用)
  const void *sendbuff;     // ★ 输入数据指针
  void *recvbuff;           // ★ 输出数据指针
  size_t count;             // ★ 总元素数

  uint64_t redOpArg;        // 归约参数 (如 1/nRanks)

  uint64_t chunkCount:25;   // 每个 chunk 的元素数
  uint64_t workCount:39;    // 此 channel 负责的元素数
  uint64_t lastChunkCount:25; // 最后一个 chunk 的元素数 (可能更小)
  uint64_t workOffset:39;   // 在全局 buffer 中的偏移
};
```

---

## 12. PTX 转译要点

如果你要将 `ncclDevKernel_AllReduce_Sum_f32_RING_LL` 的 PTX 独立运行：

### 12.1 必须构造的 host 端数据

```c
// ① ncclDevComm (device memory)
ncclDevComm devComm = {
  .rank = 0,                    // 当前 rank
  .nRanks = 2,                  // 双卡
  .node = 0,
  .nNodes = 1,
  .buffSizes = {...},           // LL/LL128/Simple 各自的 buffer 大小
  .p2pChunkSize = 65536,
  .workFifoDepth = 2048,
  .workFifoHeap = d_workFifo,   // device memory
  .abortFlag = d_abortFlag,     // device memory, 初始为 0
  .channels = d_channels,       // device memory
};

// ② ncclDevChannel (device memory, 至少 1 个)
ncclDevChannel channel = {
  .peers = d_peers,             // ncclDevChannelPeer*[] (每个 peer 一组)
  .ring = {
    .prev = 1,                  // rank 0 的 prev 是 rank 1
    .next = 1,                  // rank 0 的 next 是 rank 1 (双卡互为 prev/next)
    .userRanks = d_userRanks,
    .index = 0,
  },
  .workFifoDone = d_workFifoDone,
};

// ③ ncclWork (workFifo 中)
ncclWork work = {
  .header = {
    .funcIndex = ncclDevFuncId(ncclFuncAllReduce, ncclDevSum, ncclFloat32,
                                NCCL_ALGO_RING, NCCL_PROTO_LL),
    .isLast = 1,
    .type = ncclWorkTypeColl,
  },
  .elems[0] = {
    .isUsed = 1,
    .nWarps = 16,               // = block/32; LL 上限 512 → 16 warps (实测 f16 allreduce)
                               //   barrier 用 block=96 → nWarps=3
    .sendbuff = d_sendbuff,     // 输入数据
    .recvbuff = d_recvbuff,     // 输出数据
    .count = 65536,             // 元素个数 (实测 f16 allreduce, shape [1,16,4096])
                               //   barrier: count=1 (shape [1])
    .chunkCount = ...,          // 根据 count/nRanks/nChannels 计算
    .workCount = 65536,
    .workOffset = 0,
  },
};

// ④ 通信 buffer (peer 的 ring buffer 地址)
// 每个 peer 连接需要:
//   send: peer 的 recv buffer 地址 (通过 IPC 获取)
//   recv: 自己的 recv buffer 地址
ncclConnInfo sendConn = {
  .buffs[NCCL_PROTO_LL] = d_peer_recvbuff,  // 对方的接收 buffer
  .head = d_send_head,    // head 计数器
  .tail = d_send_tail,    // tail 计数器
  .stepSize = ...,        // 每步传输的字节数
};
```

### 12.2 CUDA Driver API 调用序列

```c
// 1. 加载 PTX
cuModuleLoadData(&module, ptx_text);
cuModuleGetFunction(&kernel, module,
    "_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LLP11ncclDevCommmP8ncclWork");

// 2. 分配 device memory
cuMemAlloc(&d_devComm, sizeof(ncclDevComm));
cuMemAlloc(&d_channel, sizeof(ncclDevChannel));
cuMemAlloc(&d_work, sizeof(ncclWork));
cuMemAlloc(&d_sendbuff, data_bytes);
cuMemAlloc(&d_recvbuff, data_bytes);
cuMemAlloc(&d_abortFlag, sizeof(uint32_t));

// 3. 拷贝数据到 device
cuMemcpyHtoD(d_devComm, &devComm, sizeof(ncclDevComm));
cuMemcpyHtoD(d_channel, &channel, sizeof(ncclDevChannel));
cuMemcpyHtoD(d_work, &work, sizeof(ncclWork));

// 4. Launch kernel  (以 f16 allreduce, 16 channel 为例)
uint64_t channelMask = 0xFFFF;  // 16 个 channel 全活跃
void* args[3] = {&d_devComm, &channelMask, &d_work};
cuLaunchKernel(kernel,
    16, 1, 1,      // grid: 16 blocks (16 channels, 数据量相关)
    512, 1, 1,     // block: 512 threads (LL 上限 NCCL_LL_MAX_NTHREADS, device.h:57)
    88416,         // shared mem: ncclShmemDynamicSize() ≈ 88416 (协议无关, sm_80)
    stream, args, NULL);
//   barrier 变体: grid=1, block=96, smem 仍 88416, count=1

// 5. 同步
cuCtxSynchronize();
```

### 12.3 注意事项

1. **ncclShmem 是静态 shared memory** — 由编译器在 `__shared__` 声明中分配，PTX 中通过 `mov.u32 %r, ncclShmem` 获取地址。转译时 layout 必须一致。

2. **ld/st.volatile.global 是通信核心** — 这些指令读写的是 peer GPU 的 ring buffer 地址，通过 NVLink/PCIe 的 P2P 映射实现跨 GPU 访问。必须正确设置 P2P 内存映射。

3. **bar.sync 的线程数** — 除了 `bar.sync 0`（全 block 同步，线程数 = block，非固定 640；LL 上限 512，barrier 调谐到 96）外，还有 `bar.sync %r241, %r14` 这种寄存器指定线程数的形式，是 NCCL 的 warp 分工机制。

4. **channelMask 与 grid 的对应** — `channelMask` 每个 bit 代表一个活跃 channel，`grid.x = popcount(channelMask)`。kernel 内部用 `__popcll` 计算 blockIdx 到 channelId 的映射。f16 allreduce `channelMask=0xFFFF, grid=16`；barrier `channelMask=0x1, grid=1`。

5. **本地构建只特化 LL collective kernel** — `libnccl.so` 里的 collective `.entry` 符号都以 `_LL` 结尾（`AllReduce_Sum_{f16,f32,…}_{RING,TREE}_LL` 等）。**LL128 / SIMPLE** 协议的 AllReduce **不走特化 kernel**，而是派发到通用 `ncclDevKernel_Generic`（通过 `ncclDevFuncTable[]` 间接调用，即 `ncclKernelMain` 的慢速路径）。所以独立跑 LL128/SIMPLE 的 PTX 时**不能复用特化 `.entry`**，得用 Generic kernel 的 PTX。
