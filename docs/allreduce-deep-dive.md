# 启动一个 NCCL device kernel 需要准备什么

> 主题：拿到 `nccl_001_ncclDevKernel_AllReduce_Sum_f16_RING_LL_...ptx` 这一份 device PTX 后，
> **要让它在双卡上正常跑起来，host / driver / runtime 侧必须先准备好哪些东西。**
>
> 这是 PTX 兼容项目的核心问题：device 函数我们能做 PTX 兼容，**但 device kernel 不是自己启动的**——
> 它依赖一整套 CPU/driver 侧的"准备工作"把数据、拓扑、通信通道、launch 参数都摆到位。
> 本文就是把这套"准备工作"拆开讲清楚，并标出**哪些必须由软件团队支持**。

---

## 0. 一句话主线

> **PyTorch 只交出 `tensor/count/dtype/op/stream`；NCCL host 侧选算法、建通道、生成"任务单"；
> driver 把 PTX 装进 GPU 并按 host 给的参数启动；device kernel 才真正跨卡搬数据 + 求和。**

device kernel（我们要做 PTX 兼容的那一份）排在**最后一步**。它能跑起来的前提是前面三步都已就绪。
所以"做 PTX 兼容"≠"只翻译 PTX"，而是要把它**赖以启动的那套 host/driver 能力**也补齐。

```
①PyTorch 交参数 ──▶ ②NCCL host 准备 ──▶ ③driver 装载+启动 ──▶ ④device kernel 执行
   (ATen)            (选算法/建comm/        (load PTX / P2P /        (我们做 PTX 兼容)
                      造 ncclWork)           launch 三参数)
        └──────────── ①②③ 全是 CPU/driver 侧，是本调研的重点 ────────────┘
```

---

## 1. 实测事实卡片（唯一真源，全文以此为准）

下面这几个数字来自**双卡 Qwen3-8B 真实推理 trace**（`nccl_ptx/call_chains.json`、`CALL_CHAINS.txt`），
不是推测。后面所有讨论都拿它当锚点。

| 项 | f16 AllReduce（MLP，主线） | f32 AllReduce（barrier 复用） |
|---|---|---|
| 实测来源 | `nccl_001_..._f16_RING_LL_...ptx` | `nccl_002_..._f32_RING_LL_...ptx` |
| 触发入口 | `dist.all_reduce()`（TP 里 MLP/Attn 输出聚合） | `dist.barrier()`（同步，复用 AllReduce） |
| algo × proto | **RING + LL** | **RING + LL** |
| **grid** | **`[1,1,1]`** | **`[1,1,1]`** |
| block | `[512,1,1]` | `[96,1,1]` |
| **dynamic smem** | **88416 B** | **88416 B** |
| count（元素数） | 4096（也见 16×4096） | 1 |
| launch API | `cudaLaunchKernelExC` | `cudaLaunchKernelExC` |
| 调用次数 / 累计耗时 | 4608 次 / 58.06 ms | 1 次 / 0.01 ms |

> ⚠️ **纠正旧版文档的三处错误**（已按实测改正，讲解时别再用旧数）：
> 1. f16 实测 **`grid=[1,1,1]`，不是 `[16,1,1]`**。本次 2 卡 / NVLink 拓扑下 NCCL 只用了 **1 个 channel**，
>    所以 `channelMask` 只有 1 个 bit、`grid.x=1`。"16 channel / channelMask=0xFFFF" 是错的。
> 2. 这份 f16 PTX 里有 **21 条** `ld/st.volatile.global`（不是 18；18 是 f32 那份）。
> 3. PTX **不是"只剩一个 entry、全部 inline"**——它仍引用两个外部符号、并保留一条间接 `call`（见 §3）。
>
> **关键观察**：f16 和 f32 两条链路，host 侧选择（RING+LL）、smem（88416）、launch API 完全一样，
> 只有 `count / block` 随数据量变。**这说明软件团队只要把"RING+LL + 这套 launch 形态"打通，
> 两条链路一起覆盖。** barrier 不是额外工作量。

---

## 2. 准备工作全景图：从冷启动到 kernel 跑起来

把"启动这一份 PTX"展开，CPU/driver 侧要做的事分**两个阶段**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 A：冷启动 / 一次性（建链）   ← 整个 communicator 生命周期只做一次       │
│                                                                            │
│  ncclGetUniqueId           生成全局唯一 ID（rank 0 产生，广播给其它 rank）   │
│  ncclCommInitRank          建 communicator：定 rank/nRanks、算 RING 拓扑     │
│       │                     （prev/next）、开 ring buffer、建跨卡 P2P 映射、 │
│       │                     在 device memory 上摆好 ncclDevComm/channels     │
│       ▼                                                                     │
│  driver: cuModuleLoadData  把 device PTX 装进 GPU、拿到 kernel 句柄          │
│  driver: P2P enable        cudaDeviceEnablePeerAccess（双卡互访 ring buffer）│
└─────────────────────────────────────────────────────────────────────────┘
                                   │  comm 建好后，每次通信只走阶段 B
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 B：每次 all_reduce（热路径）  ← 推理里每层都会触发                       │
│                                                                            │
│  ncclAllReduce(send,recv,count,dtype,op,comm,stream)   ← PyTorch 调这个      │
│       │  构造 ncclInfo                                                      │
│       ▼                                                                     │
│  ncclEnqueueCheck → taskAppend      校验、ncclSum→ncclDevSum、入队           │
│       ▼                                                                     │
│  scheduleCollTasksToPlan            ★选 RING+LL、算 channel/线程、           │
│       │                              填 ncclWork"任务单"、定 funcIndex       │
│       ▼                                                                     │
│  uploadWork                         把 ncclWork 拷到 device 的 workFifo      │
│       ▼                                                                     │
│  ncclLaunchKernel                   组 3 个参数 + grid/block/smem            │
│       │                              → cudaLaunchKernelExC                   │
│       ▼                                                                     │
│  driver: 设大 shared mem 上限 + launch  ← smem=88416 > 48KB，需 opt-in       │
│       ▼                                                                     │
│  ════════ device kernel 开始执行（我们做 PTX 兼容的部分）════════           │
└─────────────────────────────────────────────────────────────────────────┘
```

**给同事的话术**：device PTX 是"最后一棒"。它接力赛跑得起来，靠的是前面**阶段 A 建链 + 阶段 B 造任务单 + driver 装载/launch**。
我们要调研的"配套 driver/cpu 函数"，绝大多数就在这张图的阶段 A、阶段 B 和两处 `driver:` 里。

---

## 3. 这份 PTX 自己"缺"什么 —— 启动它的硬前提

很多人以为"有了 .ptx 就能 load 进去跑"。这份 dump 出来的 `.ptx` **单独喂给汇编器是装不起来的**，
原因有三，每一条都对应软件团队要补的能力：

### 3.1 它引用了两个自己没声明的外部符号

```ptx
st.shared.u32 [ncclShmem+5184], %r1;        // ① ncclShmem：静态 __shared__ 符号
...
mov.u64 %rd233, ncclDevFuncTable;           // ② ncclDevFuncTable：device global 函数表
add.s64 %rd234, %rd233, %rd232;
ld.global.u64 %rd235, [%rd234];             //   取 funcTable[funcIndex]
call %rd235, (), prototype_14;              //   ★ 间接调用 ★
```

| 外部符号 | 是什么 | 谁要提供 |
|---|---|---|
| `ncclShmem` | device 端**静态 shared memory** 块（编译器分配，kernel 按**固定偏移**读写，本 kernel 引用到 +5655 字节处）。host 在 device memory 里造的 `ncclDevComm/channel/work` 会被 warp 0/1/2 拷进它。 | device 模块里必须定义这块 shared，且 **layout 与 host 摆的结构体严丝合缝** |
| `ncclDevFuncTable` | device global，一张**函数指针表**。kernel 用 `funcIndex` 查表、`call` 过去（特化路径之外的兜底分发）。 | device 模块里必须有这张表（即：不能只 load 单个 entry，要 load 整个 NCCL device 模块） |

> **这意味着**："启动这一份 PTX"实际是"**load 整个 NCCL device 模块**"——单个 `.entry` 片段缺符号、装不起来。
> 软件团队的 PTX 加载链路要按**模块**处理，并保证 `ncclShmem` / `ncclDevFuncTable` 这类符号能被解析。

### 3.2 dump 工具把模块头剥掉了

我们的 formatter（`nccl_ptx_lib/ptx/ptx_formatter.py`）为了好读，只保留了 `.entry` 往后的部分，
**剥掉了 `.version / .target / .address_size` 模块头和 `.extern` 声明**。真正 load 时这些必须补回。
→ 软件团队拿 PTX 做兼容时，要以 **`cuobjdump -ptx` 出来的完整模块**为准，别拿 `nccl_xxx.ptx` 这种"阅读版"直接 load。

### 3.3 它要 88416 字节 dynamic shared memory（> 48KB，要 opt-in）

实测 `smem=88416`。NVIDIA 上 >48KB 的 shared memory **必须**先调
`cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 88416)` 才能 launch，否则直接报错。
→ 软件团队的 runtime 必须支持"大 shared memory opt-in"这条 driver 语义，且硬件 shared/SM 容量要够。

---

## 4. host 侧要在 device memory 里"摆好"的东西

device kernel 启动时，driver 只传 **3 个参数**进去：

```c
void* args[3] = {
    &comm->devComm,      // ① ncclDevComm*  —— 通信上下文（rank/nRanks/channels/abortFlag…）
    &plan->channelMask,  // ② uint64_t      —— 活跃 channel 位图（实测只 1 个 bit，grid.x=popcount=1）
    &plan->workHead      // ③ ncclWork*     —— "任务单" FIFO 头（send/recv buffer、count、chunk、funcIndex）
};
cudaLaunchKernelExC(&cfg /*grid=1,block=512,smem=88416,stream*/, fn, args);
```

这 3 个指针指向的内容，**全部要由 host 提前在 device memory 上构造好**，而且**字段布局必须和 PTX 里读它们的指令一致**：

| 结构体 | host 怎么来 | kernel 怎么用 | 兼容硬约束 |
|---|---|---|---|
| `ncclDevComm` | 阶段 A `ncclCommInitRank` 时建好、常驻 device | warp 0 拷进 `ncclShmem` | 字段偏移必须与 PTX 的 `ld.shared` 偏移对齐 |
| `ncclDevChannel`（含 `ring.prev/next/index`） | 同上 | warp 1 拷进 `ncclShmem` | ring 拓扑决定跨卡 send/recv 对端 |
| `ncclWork` / `ncclWorkElem` | 阶段 B 每次 `scheduleCollTasksToPlan` 现造、`uploadWork` 上传 | warp 2 拷进 `ncclShmem`，主循环消费 | `funcIndex` 要能对上 kernel 期望；buffer/count/chunk 要对 |
| ring buffer（peer 的） | 阶段 A 开辟 + P2P 映射 | `ld/st.volatile.global` 直接跨卡读写 | **必须有跨卡 P2P 内存映射** |

> **一句话总结这一节**：device kernel 不接受"普通参数"，它接受的是 **host 预先在显存里摆好的三套结构体 + 一组跨卡可见的 ring buffer**。
> 这套"摆放"动作（建 comm、造 ncclWork、上传、P2P）就是软件团队工作量的大头，详见 [ptx-compat-requirements.xlsx](ptx-compat-requirements.xlsx)。

---

## 5. driver / runtime 必须提供的能力（最该让软件团队确认的清单）

把启动这份 PTX 所需的 driver/runtime 原语单列出来（详细的"必选/可选/谁负责"在 xlsx 里）：

| 能力 | 用在哪一步 | 为什么不可省 |
|---|---|---|
| `cuModuleLoadData` / `cuModuleGetFunction` | 阶段 A 装 PTX | 没它 kernel 句柄都拿不到 |
| `cuMemAlloc` / `cuMemcpyHtoD`（cudaMalloc/Memcpy） | 摆 devComm/channel/work/buffer | device 结构体要落到显存 |
| **跨卡 P2P 内存映射** `cudaDeviceEnablePeerAccess` + IPC | 阶段 A 建 ring buffer 互访 | **Ring LL 的跨卡 `ld/st.volatile.global` 全靠它**（最关键、最易卡住的硬件/驱动能力） |
| **大 shared mem opt-in** `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | launch 前 | 88416 > 48KB，不 opt-in 直接 launch 失败 |
| `cudaLaunchKernelExC` / `cuLaunchKernel` | 阶段 B launch | 要支持 `cudaLaunchConfig_t`（grid/block/dynamicSmemBytes/stream），带 cluster/调度 attr |
| `cudaStream_t` + `cudaEvent` + stream 同步 | 全程 | NCCL 用独立 comms stream，与计算 stream 异步重叠 |
| `cuCtxSynchronize` / stream sync | 收尾 | 等 kernel 完成 |

---

## 6. device / ISA：PTX 翻译要覆盖的指令（给做翻译的同事）

这份 f16 kernel 用到的、**translate 时必须正确支持**的关键指令/语义（实测统计）：

| 指令 / 语义 | 出现次数 | 作用 | 兼容要点 |
|---|---|---|---|
| `ld/st.volatile.global`（含 `.v4.u32`） | **21** | 跨卡读写 peer ring buffer（LL 协议 data+flag 打包） | volatile 语义 + 跨卡 P2P 地址，**通信核心** |
| `bar.sync 0` / `bar.sync %r,%n` | 13 | 全 block / 指定线程数同步（NCCL warp 分工） | 线程数非固定（512 / 96 都出现） |
| `bar.warp.sync` | 10 | warp 内同步 | |
| `bar.red.popc.u32` | 1 | barrier + 跨线程 popcount 归约 | barrier 带规约语义 |
| `popc.b64` | 1 | `channelMask` 位图 → channelId 映射 | `__popcll` |
| `cvta.shared` / `cvta.to.global` | 5 | shared/global 地址空间转换 | 地址空间模型要一致 |
| `shf.r.wrap.b32` / `bfi.b64` | 6 / 4 | funnel-shift / 位域插入（LL 打包、地址计算） | |
| 间接 `call`（经 `ncclDevFuncTable`） | 1 | 非特化路径的函数表分发 | 需支持 device 端间接调用 + 函数表符号 |

> LL 协议核心：把 32-bit data + flag 打进 64-bit word，发端 `st.volatile.global.v4.u32`，
> 收端自旋 `ld.volatile.global.b32` 等 flag——这就是那 21 条 volatile 的用途。

---

## 7. 落地策略：两条路，工作量不同

| | 策略 1：只换 device（复用 NCCL host） | 策略 2：host 也自研（drop-in 兼容库） |
|---|---|---|
| 做法 | NCCL host 全留，只让软件团队补**阶段 A/B 之外的** runtime（load/P2P/launch/大 smem）+ 翻译 device PTX | 自己实现 `ncclAllReduce` + comm 初始化 + 枚举，内部自造 `ncclWork`，host 全自研 |
| 工作量 | **最小**：阶段 A/B 的 host 逻辑白送 | 大：要复刻整套 enqueue/schedule/launch |
| 风险 | 要求 runtime 能被 NCCL host 当 CUDA runtime 用（driver API 兼容度高） | 要自己保证 device 结构体 ABI 与 PTX 严丝合缝 |
| 共同硬约束 | **不管哪条路**：①`ncclDevComm/ncclWork/channelMask` 布局必须和 PTX 一致；②跨卡 P2P；③88416 大 shared mem；④load 整个 device 模块（解析 `ncclShmem`/`ncclDevFuncTable`） |

> 建议：**先按策略 1 打通**（白送 host 逻辑，验证 P2P + 大 smem + launch + 翻译这条最短路），再评估是否需要策略 2。

---

## 8. 一句话回答"配套 driver/cpu 函数怎么支持"

> device PTX 兼容只是最后一棒。让它跑起来，软件团队要补的是：
> **① 建链（comm 初始化 + 跨卡 P2P + ring buffer）、② 每次通信造"任务单"ncclWork 并上传、
> ③ driver 装载整个 device 模块、按 3 参数 + grid/block/88416 大 shared mem 启动、④ stream/同步。**
> 这些就是 [ptx-compat-requirements.xlsx](ptx-compat-requirements.xlsx) 里逐条列出、标了"必选 / 谁负责"的内容。
> 一页纸讲解稿见 [ptx-compat-onepager.md](ptx-compat-onepager.md)。

---
---

# 附录（硬核细节，按需翻阅）

> 正文到此结束。下面是完整 9 层调用链、数据结构逐字段、转译落地代码——讲解时**不用展开**，
> 需要查证某一层时再翻。所有内容已对照实测 trace 校正。

## 附录 A：完整 9 层调用链（torch → device）

```
Python:  dist.all_reduce(tensor, op=ReduceOp.SUM)
  ▼ Layer 1  torch.distributed.all_reduce()                      [Python]
  ▼ Layer 2  ProcessGroupNCCL::allreduce() → allreduce_impl()    [C++]
  ▼ Layer 3  collective() 模板：取 comm/stream、建 Work、调 fn   [C++]
  ▼ Layer 4  ncclAllReduce(send,recv,count,dtype,op,comm,stream) [NCCL C API] → 构造 ncclInfo
  ▼ Layer 5  ncclEnqueueCheck → taskAppend                       [入队] hostToDevRedOp: ncclSum→ncclDevSum
  ▼ Layer 6  scheduleCollTasksToPlan                             [调度] topoGetAlgoInfo 选 RING+LL；
  │            ncclDevFuncId 定 funcIndex；initCollWorkElem 填 ncclWork；uploadWork 上传
  ▼ Layer 7  ncclLaunchKernel → cudaLaunchKernelExC              [launch] 3 参数 + grid/block/smem
  ▼ Layer 8  ncclDevKernel_AllReduce_Sum_f16_RING_LL            [__global__] = ncclKernelMain<…>()
  ▼ Layer 9  RunWork<AllReduce,f16,Sum,RING,LL>::run() → runRing<ProtoLL>   [device]
             Primitives: send → recvReduceSend → … → directRecv
```

### barrier 变体（同一台机器，不同入口）

NCCL **没有** `ncclBarrier` collective。`dist.barrier()` 在缓存的 1 元素 tensor 上跑一次
`ncclAllReduce(ncclFuncAllReduce)`，所以 **Layer 4–9 与 `all_reduce` 完全相同**，只有入口两层不同：

```
Layer 1  torch.distributed.barrier()                 (distributed_c10d.py:4122)
Layer 2  ProcessGroupNCCL::barrier() → allreduce_impl(barrierTensor_) → 接回 Layer 4
```
实测就是那条 `f32_RING_LL`、`count=1`、`block=96` 的链路。默认 used-only 会把它排除，加
`--include-sync-kernels` 才纳入（见 `docs/adr/003-call-chain-tracing.md`）。

---

## 附录 B：关键源码片段（逐层）

### B.1 Layer 2 — ProcessGroupNCCL::allreduce / allreduce_impl
来源：`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
```cpp
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) tensor = at::view_as_real(tensor);   // 复数→实数视图
  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);     // 节点内 SHMEM 快速路径
    if (algo != AllReduceAlgo::NONE) { intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>(); }
  }
  return allreduce_impl(tensor, opts);                           // 进入 NCCL 路径
}

// allreduce_impl: 用 collective() 包一个调 ncclAllReduce 的 lambda（in-place: input==output）
return collective(tensor, tensor,
    [&](at::Tensor& in, at::Tensor& out, ncclComm_t comm, at::cuda::CUDAStream& s) {
      auto dt = getNcclDataType(in.scalar_type());
      auto op = getNcclReduceOp(opts.reduceOp, in, dt, comm);
      return ncclAllReduce(in.data_ptr(), out.data_ptr(), in.numel(), dt, op, comm, s.stream());
    }, OpType::ALLREDUCE, "nccl:all_reduce");
```

类型 / 归约映射（torch → NCCL）：

| PyTorch ScalarType | NCCL ncclDataType_t | 字节 | | PyTorch ReduceOp | NCCL ncclRedOp_t |
|---|---|---|---|---|---|
| kHalf (f16) | ncclFloat16 | 2 | | SUM | ncclSum |
| kFloat (f32) | ncclFloat32 | 4 | | AVG | ncclAvg (2.10+) |
| kBFloat16 | ncclBfloat16 | 2 | | PRODUCT | ncclProd |
| kDouble | ncclFloat64 | 8 | | MIN / MAX | ncclMin / ncclMax |
| kInt/kLong | ncclInt32/ncclInt64 | 4/8 | | PREMUL_SUM | ncclRedOpCreatePreMulSum |

特殊：`bool + SUM → ncclMax`（避免 uint8 溢出）。

### B.2 Layer 4 — ncclAllReduce
来源：`nccl/src/collectives.cc`
```c
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff,
    count, datatype, op, 0 /*root*/, comm, stream,
    ALLREDUCE_CHUNKSTEPS /*=NCCL_STEPS/2=4*/, ALLREDUCE_SLICESTEPS /*=NCCL_STEPS/4=2*/ };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}
```
注意 `count` 是**元素数**，不是字节数；`sendbuff==recvbuff` 即 in-place。

### B.3 Layer 5 — taskAppend + hostToDevRedOp
来源：`nccl/src/enqueue.cc`
```c
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  hostToDevRedOp(&info->opFull, info->op, info->datatype, comm);  // ncclSum→ncclDevSum
  if (comm->nRanks == 1) {                                        // 单 rank：直接 memcpy，不 launch
    ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count,
                      info->opFull, info->datatype, info->stream); return ncclSuccess;
  }
  ncclGroupCommJoin(info->comm);                                  // 多 rank：入 collQueue
  struct ncclInfo* t = ncclMemoryStackAlloc<struct ncclInfo>(&comm->memScoped);
  info->algorithm = NCCL_ALGO_UNDEF; info->protocol = NCCL_PROTO_UNDEF;  // 算法待选
  memcpy(t, info, sizeof(*info));
  ncclIntruQueueSortEnqueue(&tasks->collQueue, t, collCmp);
}
```
归约转换表：

| 用户 op | 设备 op | 说明 |
|---|---|---|
| ncclSum | ncclDevSum | 直接求和 |
| ncclProd | ncclDevProd | 直接求积 |
| ncclMin/Max | ncclDevMinMax | XOR 符号位后求 min |
| ncclAvg(float) | ncclDevPreMulSum | 各 rank 先乘 1/nRanks 再求和 |
| ncclAvg(int) | ncclDevSumPostDiv | 求和后除以 nRanks |

### B.4 Layer 6 — scheduleCollTasksToPlan（算法选择 + funcIndex）
来源：`nccl/src/enqueue.cc:743-870`
```c
// ① 选 algo×proto：topoGetAlgoInfo 暴力遍历所有组合，对每个调 ncclTopoGetAlgoTime 估时取最小
//    （不是按数据量查表！实测 2 卡/NVLink 下连 4 字节也选 RING+LL）
getTunerInfo(collInfo) 或 topoGetAlgoInfo(collInfo);
// ② channel / 线程数
getChannnelThreadInfo(collInfo);
// ③ kernel 函数索引
collInfo->workFuncIndex = ncclDevFuncId(ncclFuncAllReduce, ncclDevSum, ncclFloat16,
                                        NCCL_ALGO_RING, NCCL_PROTO_LL);
// ④ pattern：AllReduce+RING → ncclPatternRingTwice
// ⑤ computeCollChunkInfo  ⑥ initCollWorkElem 填 ncclWork  ⑦ plan->kernelFn = ncclDevKernelForFunc[idx]
```

`ncclDevFuncId` 线性索引（`device.h`）：
```
row = ((devRedOp * NumTypes + type) * nAlgos + algo) * NCCL_NUM_PROTOCOLS + proto
    NumTypes=10, nAlgos=6, NCCL_NUM_PROTOCOLS=3
例 AllReduce+Sum+f32+RING+LL: devRedOp=0,type=7,algo=1,proto=0
   row = ((0*10+7)*6+1)*3+0 = 129 → funcIndex = ncclDevFuncRowToId[129]
```

> 选择器实测结论：本地 `libnccl.so` 编译进了 `AllReduce_Sum_{u8,f16,f32,f64,u32,u64,bf16}_{RING,TREE}_LL`
> 共 14 个 LL 特化 kernel，但 2 卡/NVLink 下 `topoGetAlgoInfo` **只选中 RING+LL**——TREE 编译了没被选。
> **LL128 / SIMPLE 不走特化 kernel**，而是经 `ncclDevFuncTable[]` 派发到通用 `ncclDevKernel_Generic`
> （即正文 §3.1 那条间接 `call`）。所以独立跑 LL128/SIMPLE 不能复用 LL 的特化 entry。

### B.5 Layer 7 — ncclLaunchKernel
来源：`nccl/src/enqueue.cc:1351-1409`
```c
void* fn = plan->kernelFn;                                  // = ncclDevKernel_..._f16_RING_LL
dim3 grid  = {(unsigned)plan->channelCount, 1, 1};          // 实测=1（只 1 个 channel）
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};        // LL≤512，按数据量调谐，最低 3 warp=96
size_t smem = ncclShmemDynamicSize(comm->cudaArch);         // 协议无关取 max，sm_80≈88416（实测）
void* args[3] = {&comm->devComm, &plan->channelMask, &plan->workHead};
#if CUDART_VERSION >= 11080
  cudaLaunchConfig_t cfg = {grid, block, smem, launchStream, /*attrs: cluster/policy/memsync*/};
  cudaLaunchKernelExC(&cfg, fn, args);
#else
  cudaLaunchKernel(fn, grid, block, args, smem, launchStream);
#endif
```
`smem` 为什么 LL 也非 0：`ncclShmemDynamicSize` 取**所有协议的 max**（SIMPLE/NVLS 的 scratch 最大），
对含 LL 特化在内的所有 kernel 都返回非零 ≈88416（`device.h:395-408`）。"LL→smem=0" 是错的。

### B.6 Layer 8 — ncclKernelMain（device 入口）
来源：`nccl/src/device/common.h:124-213`
```c
__global__ void ncclDevKernel_AllReduce_Sum_f16_RING_LL(
    ncclDevComm* comm, uint64_t channelMask, ncclWork* workHead) {
  ncclKernelMain<SpecializedFnId,
      RunWork<ncclFuncAllReduce, half, FuncSum<half>, NCCL_ALGO_RING, NCCL_PROTO_LL>>(
      comm, channelMask, workHead);
}
// ncclKernelMain 内部：
//  阶段1 blockIdx→channelId：tid<32 时 __popcll(channelMask & ((1<<tid)-1)) 找第几个活跃 channel
//  阶段2 warp0/1/2 分别 copyToShmem16 加载 comm/channel/work 到 ncclShmem
//  阶段3 __syncthreads；主循环处理 work 链表
//  阶段4 funcIndex==SpecializedFnId 走特化 run()；否则 ncclDevFuncTable[funcIndex]() ← 间接 call
//  阶段5 检查 isLast / abortFlag，取下一个 work 或退出
```

### B.7 Layer 9 — runRing<ProtoLL>
来源：`nccl/src/device/all_reduce.h`
```c
template<typename T, typename RedOp, typename Proto>
__device__ void runRing(ncclWorkElem *args) {
  Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
      tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);
  // Ring AllReduce：总步数 2*(nranks-1)
  //   Reduce-Scatter：send → (recvReduceSend ×) → directRecvReduceCopySend(postOp)
  //   AllGather：(directRecvCopySend ×) → directRecv
}
```
LL 协议打包（那 21 条 volatile 的来历）：
```
发送 st.volatile.global.v4.u32 [addr], {data_lo, flag, data_hi, flag}
接收 自旋 ld.volatile.global.b32 flag；命中后 ld.volatile.global.v4.u32 取 data
```

---

## 附录 C：关键数据结构（逐字段，做 ABI 兼容时对照）

### ncclDevComm
```c
struct ncclDevComm {
  int rank, nRanks, node, nNodes;
  int buffSizes[NCCL_NUM_PROTOCOLS];   // 每协议 ring buffer 大小
  int p2pChunkSize;
  int workFifoDepth;  struct ncclWork* workFifoHeap;     // work FIFO（device memory）
  int* collNetDenseToUserRank;
  volatile uint32_t* abortFlag;                          // host 可写的中断标志
  struct ncclDevChannel* channels;                       // channel 数组
};
```
### ncclDevChannel
```c
struct ncclDevChannel {
  struct ncclDevChannelPeer** peers;
  struct ncclRing ring;   // .prev .next .userRanks .index  ← 双卡互为 prev/next
  struct ncclTree tree;   // .depth .up .down[3]
  struct ncclTree collnetChain;  struct ncclDirect collnetDirect;  struct ncclNvls nvls;
  uint32_t* workFifoDone;
};
```
### ncclWork（512B）/ ncclWorkElem
```c
struct ncclWork {
  struct ncclWorkHeader header;  // funcIndex / isLast / inFifo / type / workNext
  union { struct ncclWorkElem elems[9];
          struct ncclWorkElemP2p p2pElems[16];
          struct ncclWorkElemReg regElems[2]; };
};
struct ncclWorkElem {
  uint8_t isUsed:1, redOpArgIsPtr:1, oneNode:1; uint8_t regUsed, nWarps, direct;
  uint32_t root;
  const void* sendbuff; void* recvbuff; size_t count;   // ★ 三大要素
  uint64_t redOpArg;
  uint64_t chunkCount:25, workCount:39, lastChunkCount:25, workOffset:39;
};
```

---

## 附录 D：把这份 PTX 独立跑起来的 driver 调用序列（参考）

> 仅作"端到端要做哪些 driver 调用"的参考；真实落地建议复用 NCCL host（策略 1）。

```c
// 1. 加载整个 device 模块（不是单个 entry！要带 ncclShmem / ncclDevFuncTable）
cuModuleLoadData(&module, ptx_module_text);                 // 用 cuobjdump -ptx 的完整模块
cuModuleGetFunction(&kernel, module,
    "_Z39ncclDevKernel_AllReduce_Sum_f16_RING_LLP11ncclDevCommmP8ncclWork");

// 2. 阶段 A：建 comm（含跨卡 P2P）→ 在 device memory 上摆好 devComm/channels/ring buffer
//    （此处最重，建议直接复用 ncclCommInitRank）
cudaDeviceEnablePeerAccess(peer, 0);                        // 双卡互访 ring buffer

// 3. 阶段 B：造 ncclWork、上传 workFifo（funcIndex 对上、buffer/count/chunk 对上）

// 4. 开大 shared memory 上限（88416 > 48KB，必须）
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 88416);

// 5. launch：3 参数 + 实测 grid/block/smem
uint64_t channelMask = 0x1;                                 // 实测只 1 个 channel
void* args[3] = {&d_devComm, &channelMask, &d_workHead};
cudaLaunchKernelExC(&{ .gridDim={1,1,1}, .blockDim={512,1,1},
                       .dynamicSmemBytes=88416, .stream=stream }, kernel, args);
//   barrier 变体：block={96,1,1}，count=1，其余相同

// 6. 同步
cuCtxSynchronize();
```

转译注意事项：
1. `ncclShmem` 是静态 shared，PTX 按固定偏移访问，翻译后 layout 必须一致。
2. `ld/st.volatile.global` 读写 peer ring buffer，依赖 P2P 映射，volatile 语义不能丢。
3. `bar.sync` 线程数非固定（512/96 都有），还有 `bar.sync %r,%n` 寄存器指定形式。
4. `channelMask`/`grid` 对应：`grid.x = popcount(channelMask)`，本次实测都是 1。
5. LL128/SIMPLE 不复用 LL 特化 entry，走 `ncclDevKernel_Generic`（经 `ncclDevFuncTable` 间接 call）。
