# PTX 兼容一页纸 · 讲解速记

> 给同事开会讲用。一句话主线 + 一张准备流程图 + SW 团队要做的事清单 + 实测数字 + 常见追问。
> 详表见 [ptx-compat-requirements.xlsx](ptx-compat-requirements.xlsx)，原理见 [allreduce-deep-dive.md](allreduce-deep-dive.md)。

---

## 1. 一句话（先抛这句，立住全场）

> **device 函数我们能做 PTX 兼容；但 device kernel 不会自己启动。**
> **PyTorch 交参数 → NCCL host 选算法 + 造"任务单" → driver 装 PTX + 开跨卡 P2P + launch → device kernel 才执行。**
> **前三步全是 CPU/driver 侧——这就是软件团队要支持的"配套 driver/cpu 函数"。**

调研结论一句话：**device PTX 兼容只是最后一棒；让它跑起来的工作量大头在 host/driver。**

---

## 2. 一张图（准备一份 PTX 启动起来的全过程）

```
   ① PyTorch              ② NCCL host(CPU)            ③ driver / runtime          ④ device kernel
  ─────────────         ──────────────────          ────────────────────        ───────────────
  dist.all_reduce  ──▶  ncclAllReduce            ──▶ cuModuleLoadData(装整模块) ─▶ ncclDevKernel_
  (tensor,count,        ↓                            cudaDeviceEnablePeerAccess     AllReduce_Sum_
   dtype,op,stream)     选 RING+LL(topoGetAlgo)      (跨卡 P2P ring buffer)         f16_RING_LL
                        造 ncclWork 任务单            cudaFuncSetAttribute           ↓
                        (sendbuff/count/funcIdx)     (开 88416B 大 shared mem)      跨卡 ld/st.
                        uploadWork 到 workFifo        cudaLaunchKernelExC            volatile.global
                        组 3 参数:                     (grid=1,block=512,smem=88416)  + 求和
                        {devComm,channelMask,workHead}                              ↓
                                                                                   结果写回
   └── 都是 CPU/driver 侧，软件团队要补 ──────────────────────┘     └ 我们做 PTX 兼容 ┘
```

**记三件事**：
1. device kernel 只收 **3 个指针参数**：`{ncclDevComm*, channelMask, ncclWork*}`——内容全要 host 提前在显存摆好。
2. 这 3 套结构体的**字段布局必须和 PTX 读它们的指令对齐**（ABI 硬约束）。
3. 跨卡通信靠 **P2P ring buffer + volatile 读写**，不是普通 kernel 参数传数据。

---

## 3. SW 团队要做的事（按"谁负责"分，照着派活）

| # | 团队 | 要做什么 | 难度 | 不做会怎样 |
|---|---|---|---|---|
| 1 | **驱动** | 跨卡 **P2P 内存映射**（`cudaDeviceEnablePeerAccess`+IPC） | ★★★ | Ring LL 跨卡读写失败，通信跑不起来 |
| 2 | **驱动** | **大 shared mem opt-in**（`cudaFuncSetAttribute` 88416B，>48KB 必须） | ★★★ | launch 直接报错 |
| 3 | **驱动** | 装**整个** device 模块（`cuModuleLoadData`）+ 取句柄 + `cudaLaunchKernelExC` | ★★ | 装不进/启不动 |
| 4 | **驱动** | 显存分配/拷贝、stream/event/同步 | ★/★★ | 结构体无处放、无法重叠/收尾 |
| 5 | **软件(host)** | `ncclAllReduce` + comm 初始化 + datatype/redop 枚举（对接 PyTorch） | ★~★★ | torch c10d 路径接不上 |
| 6 | **软件(host)** | enqueue→schedule→造 ncclWork→launch（**复用 NCCL 则白送**） | ★★★ | 选不出算法、产不出任务单 |
| 7 | **翻译(device)** | 解析外部符号 `ncclShmem`/`ncclDevFuncTable`；翻译 volatile.global / bar.* / popc / cvta 等 ISA | ★★★ | 装载失败 / 通信结果错 |
| 8 | **全体** | 保证 `ncclDevComm/ncclWork/channelMask` 布局与 PTX **严丝合缝**（ABI） | ★★★ | 行为未定义、难 debug |

> **建议落地顺序（策略 1，工作量最小）**：复用 NCCL host（#6 白送）→ 先打通 #1 P2P + #2 大 smem + #3 装载/launch + #7 翻译 → 验证 f16 RING+LL 跑通 → 再看要不要自研 host（策略 2）。

---

## 4. 实测数字（被问到就甩这张，全部来自双卡 Qwen3-8B trace）

| | f16 all_reduce（主线/热点） | f32 barrier（复用 AllReduce） |
|---|---|---|
| algo×proto | **RING + LL** | **RING + LL** |
| grid / block | **[1,1,1]** / [512,1,1] | **[1,1,1]** / [96,1,1] |
| dynamic smem | **88416 B** | **88416 B** |
| count | 4096 | 1 |
| 次数 / 耗时 | 4608 / 58 ms | 1 / 0.01 ms |

**一句话点评**：两条链路 host 选择、smem、launch 形态**完全一样**，只有 `count/block` 变 →
**打通 RING+LL 一条，all_reduce 和 barrier 同时覆盖**，barrier 不是额外活。

---

## 5. 常见追问（提前备好答案）

**Q：有了 .ptx 不就能直接 load 跑？**
A：不能。这份 .ptx 引用了两个它自己**没声明**的符号——`ncclShmem`（静态 shared）和 `ncclDevFuncTable`
（device 函数表，里面还有一条间接 `call`）。必须 load **整个 NCCL device 模块**才能解析。我们 dump 出来的是"阅读版"，
还被剥了模块头，真正 load 要用 `cuobjdump -ptx` 的完整模块。

**Q：barrier 要不要单独做？**
A：不用。NCCL 没有 `ncclBarrier`，`dist.barrier()` 就是在 1 元素 tensor 上跑一次 AllReduce。同一套链路。

**Q：为什么 grid 是 1 不是 16？**
A：2 卡 / NVLink 拓扑下 NCCL 只用了 1 个 channel，`channelMask` 只有 1 个 bit，`grid.x=popcount=1`。
（旧文档写 16 是错的，已改正。）

**Q：LL 协议 shared memory 不是 0 吗？**
A：不是。`ncclShmemDynamicSize` 对所有协议取 max，实测 LL 特化 kernel 也要 **88416B**，且 >48KB 必须 opt-in。

**Q：最容易卡在哪？**
A：① 跨卡 **P2P**（通信地基）；② **88416B 大 shared mem** 的 driver/硬件支持；③ **结构体 ABI** 对不齐（最难 debug）。

**Q：LL128 / SIMPLE 协议呢？**
A：本地只特化了 **LL** kernel。LL128/SIMPLE 不走特化 entry，经 `ncclDevFuncTable` 派发到通用
`ncclDevKernel_Generic`。先做 LL 主线，其余按需。

---

## 6. 一句话收尾

> 我们要支持的"配套 driver/cpu 函数"= **建链(comm+P2P) + 造任务单(ncclWork) + 装载整模块 + 大 shared mem + launch/同步**。
> device PTX 翻译是最后一棒；先按"复用 NCCL host"打通 RING+LL，就能同时跑通 all_reduce 和 barrier 两条真实链路。
