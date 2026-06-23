#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 AllReduce 全链路 PTX 兼容依赖表 (sunrise 国产卡) 的 Excel。
依据: docs/allreduce-deep-dive.md + nccl_ptx/CALL_CHAINS.txt 实测 trace。
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from pathlib import Path

wb = Workbook()

# ============================================================
# Sheet 0: 新手导图
# ============================================================
ws = wb.active
ws.title = "先读导图"
ws.append(["主题", "新手版解释", "对应表 / 文档"])
guide_rows = [
    ("一句话",
     "PyTorch 发起 all_reduce；NCCL host 侧选择算法并准备任务单；GPU device kernel 负责跨卡搬数据和归约。",
     "docs/allreduce-deep-dive.md §给 NCCL 小白的速读版"),
    ("先看哪张表",
     "先读本页，再看“三层依赖表”判断要实现哪些 host/runtime 能力；最后看“④device层_ISA要求”和“验证检查清单”。",
     "本工作簿"),
    ("三段心智模型",
     "① ATen/PyTorch 负责入口；② NCCL/pccl 负责 API、调度、构造 ncclWork；③ Runtime 负责 launch 和跨卡内存；④ device PTX 真正执行。",
     "三层依赖表 / ④device层_ISA要求"),
    ("barrier 为什么也出现",
     "NCCL 没有独立 ncclBarrier；PyTorch barrier 用 1 元素 tensor 跑一次 AllReduce，所以会看到 AllReduce kernel。",
     "allreduce-deep-dive.md §barrier 变体"),
    ("实测主线",
     "双卡 Qwen3-8B trace 里，f16 all_reduce 和 f32 barrier 都选中 RING+LL；差异主要是 count/grid/block。",
     "落地策略与约束 / allreduce-deep-dive.md §7.4"),
    ("最容易踩坑",
     "不是所有 NCCL 内部函数都必须照抄，但最终写给 device 的结构体布局、channelMask、P2P ring buffer、smem/block 必须和 PTX 期待一致。",
     "验证检查清单"),
]
for r in guide_rows:
    ws.append(list(r))

# ============================================================
# Sheet 1: 三层依赖表 (主表)
# ============================================================
ws = wb.create_sheet("三层依赖表")
ws.title = "三层依赖表"

headers = ["层", "子类", "函数 / 符号", "出处", "作用", "sunrise 侧需提供", "必选/可选"]
ws.append(headers)

# (层, 子类, 函数, 出处, 作用, sunrise要求, 必选/可选)
rows = [
    # ① ATen
    ("① ATen\n(torch算子分发)", "torch op",
     "c10d::allreduce_",
     "trace[aten] / doc L2",
     "all_reduce 分发:校验单 tensor、复数→view_as_real、节点内 SHMEM 快速路径否则走 NCCL",
     "sunrise torch 构建需注册此 op,路由到 pccl",
     "必选"),
    ("① ATen\n(torch算子分发)", "torch op",
     "c10d::barrier",
     "trace[aten] / doc §barrier变体",
     "barrier 分发;内部 allreduce_impl(barrierTensor_) 复用 AllReduce 机器",
     "同上,路由到 pccl(走 allreduce)",
     "必选"),
    ("① ATen\n(torch算子分发)", "torch op",
     "record_param_comms",
     "trace[aten]",
     "param↔comms 映射,profiling/DTENSOR 用",
     "仅为 trace 完整,非功能路径",
     "可选"),

    # ② NCCL → pccl (公共 API)
    ("② NCCL→pccl\n(链接库)", "公共API(必须导出)",
     "ncclAllReduce(sendbuff,recvbuff,count,datatype,op,comm,stream)",
     "trace[runtime] / doc L4",
     "NCCL C API 入口;构造 ncclInfo 后调 ncclEnqueueCheck",
     "pccl 同名同签名实现",
     "必选"),
    ("② NCCL→pccl\n(链接库)", "公共API(必须导出)",
     "ncclGetUniqueId / ncclCommInitRank / ncclCommInitAll",
     "doc 12.1(前置)",
     "communicator 初始化,建 rank/nRanks/channel 拓扑",
     "pccl 必须提供(构造等价 devComm)",
     "必选"),
    ("② NCCL→pccl\n(链接库)", "公共API(必须导出)",
     "枚举 ncclDataType_t / ncclRedOp_t / ncclComm_t",
     "doc 3.3",
     "类型与归约操作枚举(torch 侧 getNcclDataType/getNcclReduceOp 依赖其值)",
     "pccl 需兼容这套枚举值",
     "必选"),

    # ② NCCL → pccl (host 内部)
    ("② NCCL→pccl\n(链接库)", "host内部(自研等价/复用白送)",
     "ncclEnqueueCheck",
     "trace[runtime] / doc L5",
     "group 语义、comm/参数校验、调 taskAppend",
     "pccl 内部等价",
     "必选*"),
    ("② NCCL→pccl\n(链接库)", "host内部(自研等价/复用白送)",
     "taskAppend (+hostToDevRedOp)",
     "trace[runtime] / doc L5",
     "ncclSum→ncclDevSum 转换;单 rank memcpy 快速路径;多 rank 入 collQueue",
     "pccl 内部等价",
     "必选*"),
    ("② NCCL→pccl\n(链接库)", "host内部(自研等价/复用白送)",
     "scheduleCollTasksToPlan (+topoGetAlgoInfo/ncclDevFuncId/initCollWorkElem/uploadWork/getChannnelThreadInfo/computeCollChunkInfo/getPatternInfo)",
     "trace[runtime] / doc L6",
     "选 algo×proto(实测 RING+LL)、算 channel/线程、填 ncclWorkElem、上传 workFifo",
     "pccl 内部等价;必须产出与 PTX 期望一致的 devComm/ncclWork/channelMask 布局",
     "必选*"),
    ("② NCCL→pccl\n(链接库)", "host内部(自研等价/复用白送)",
     "ncclLaunchKernel (+ncclShmemDynamicSize)",
     "trace[runtime] / doc L7",
     "设 grid/block/smem、组 3 参数 {&devComm,&channelMask,&workHead}、调 cudaLaunchKernelExC",
     "pccl 内部等价",
     "必选*"),
    ("② NCCL→pccl\n(链接库)", "host内部(自研等价/复用白送)",
     "ncclGroupStart / ncclGroupEnd",
     "doc L5",
     "group 批提交语义",
     "pccl 内部等价",
     "必选*"),

    # ③ Runtime
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cudaLaunchKernelExC / cudaLaunchKernel",
     "trace[launch] / doc L7",
     "启动 kernel(含 cudaLaunchConfig_t:grid/block/dynamicSmemBytes/stream)",
     "sunrise runtime 必须支持",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cudaStream_t + cudaEvent + syncStream",
     "trace[runtime] / doc L3",
     "专用 comms stream、与计算 stream 异步同步",
     "sunrise runtime 必须支持",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cuModuleLoadData + cuModuleGetFunction",
     "doc 12.2",
     "加载转译后的 PTX、取 kernel 句柄",
     "sunrise runtime 必须支持(PTX/SASS 加载)",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cuLaunchKernel",
     "doc 12.2",
     "driver 级 launch",
     "sunrise runtime 必须支持",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cuMemAlloc / cuMemcpyHtoD (cudaMalloc/cudaMemcpy)",
     "doc 12.2",
     "分配并填充 devComm/channel/work/sendbuff/recvbuff/abortFlag",
     "sunrise runtime 必须支持",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "P2P: cudaDeviceEnablePeerAccess / peer memory mapping",
     "doc 12.3 注②",
     "跨 GPU ring buffer 的 ld/st.volatile.global 读写(NVLink/PCIe P2P)",
     "sunrise 硬件/驱动必须支持跨卡 P2P 内存映射(关键)",
     "必选(关键)"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)",
     "doc L7/12.2",
     "申请 88416 B 动态 shared mem",
     "sunrise runtime 必须支持",
     "必选"),
    ("③ Runtime\n(sunrise runtime)", "CUDA runtime/driver",
     "cudaCtxSynchronize / stream sync",
     "doc 12.2",
     "同步收尾",
     "sunrise runtime 必须支持",
     "必选"),
]
for r in rows:
    ws.append(list(r))

# ============================================================
# Sheet 2: 第④层 device (翻译目标, ISA 要求)
# ============================================================
ws2 = wb.create_sheet("④device层_ISA要求")
ws2.append(["项", "内容", "出处", "说明"])
dev_rows = [
    ("翻译目标",
     ".visible .entry ncclDevKernel_AllReduce_Sum_{f16,f32}_RING_LL",
     "nccl_ptx/*.ptx / doc L8",
     "已 inline 了 ncclKernelMain / RunWork::run / runRing / ProtoLL,PTX 里只剩此一个 entry"),
    ("ISA要求", "ld/st.volatile.global (跨卡 P2P)",
     "doc 10.5/12.3 注②",
     "读写 peer GPU 的 ring buffer;对应第③层 P2P 映射"),
    ("ISA要求", "bar.sync 0 + bar.sync %r,%n (warp 分工形式)",
     "doc 9.2/12.3 注③",
     "线程数非固定(LL 上限 512,barrier 调谐到 96)"),
    ("ISA要求", "__popcll",
     "doc 9.2 阶段1",
     "channelMask(64-bit 位图)→channelId 映射"),
    ("ABI要求", "静态 ncclShmem layout 二进制兼容",
     "doc 12.3 注①",
     "host 填的 ncclDevComm/ncclWork 结构体布局必须与 PTX 读取一致"),
]
for r in dev_rows:
    ws2.append(list(r))

# ============================================================
# Sheet 3: 落地策略 & 跨层约束
# ============================================================
ws3 = wb.create_sheet("落地策略与约束")
ws3.append(["项", "内容"])
strat_rows = [
    ("策略1: Drop-in 替换",
     "pccl 只实现第②层公共 API(ncclAllReduce+ncclComm*+枚举),内部自研。"
     "ncclEnqueueCheck~ncclLaunchKernel 不用逐个照抄(表中标*的),只要最终产出布局兼容的 devComm/ncclWork 并 launch。"
     "工作量小,但要自己保证 device 结构体 ABI 与 PTX 一致。"),
    ("策略2: 复用 NCCL host + 换 device PTX",
     "host 全留(第②层内部白送,标*的全部白送),sunrise 只补第③层 runtime 和第④层 PTX 转译。"
     "工作量最小,但要求 sunrise runtime 能被 NCCL host 当 CUDA runtime 用。"),
    ("跨层硬约束1",
     "第②层填进 device memory 的 ncclDevComm/ncclWork/channelMask 布局,必须和第④层 PTX 里读这些结构的指令严丝合缝。"),
    ("跨层硬约束2",
     "barrier 路径与 all_reduce 共用第②③④层,只是入口 aten op 不同(c10d::barrier vs c10d::allreduce_)。"),
    ("实测依据",
     "双卡 Qwen3-8B trace:① f16 RING_LL (MLP all_reduce, shape=[1,16,4096], grid=[16,1,1], block=[512,1,1], smem=88416);"
     "② f32 RING_LL (barrier, count=1, grid=[1,1,1], block=[96,1,1], smem=88416)。本地构建只特化了 LL。"),
    ("图例",
     "必选=功能路径必需;可选=仅 trace/profiling;必选*=Drop-in 策略下 pccl 自研需等价实现,复用 NCCL host 策略下白送。"),
]
for r in strat_rows:
    ws3.append(list(r))

# ============================================================
# Sheet 4: 术语表
# ============================================================
ws4 = wb.create_sheet("术语表")
ws4.append(["术语", "一句话理解", "为什么重要"])
term_rows = [
    ("AllReduce",
     "每张 GPU 先各有一份数据；调用后，每张 GPU 都拿到所有 rank 归约后的完整结果。",
     "训练里常用于梯度求和/平均，是本次 PTX 链路的 collective 类型。"),
    ("rank",
     "通信组里的 GPU 编号；双卡时通常是 rank 0 和 rank 1。",
     "ring/tree 拓扑、prev/next、buffer 地址都按 rank 组织。"),
    ("communicator / ncclComm",
     "一组 GPU 的通信上下文，记录 rank 数、拓扑、channel、device 侧结构体地址等。",
     "PyTorch 每次 all_reduce 最终都要拿到一个 comm 才能调用 ncclAllReduce。"),
    ("channel",
     "NCCL 把一次大通信拆成多条并行“车道”。",
     "实测 f16 all_reduce 用 16 个 channel，所以 grid.x=16；barrier 只用 1 个。"),
    ("algo",
     "通信拓扑算法，例如 RING、TREE、NVLS。",
     "algo 决定 device 端走 runRing 还是 runTree；本次实测选中 RING。"),
    ("proto",
     "通信协议，例如 LL、LL128、SIMPLE。",
     "proto 决定底层搬运方式；本地 used-only PTX 是 LL 特化 kernel。"),
    ("ncclWork / ncclWorkElem",
     "host 写给 device kernel 的任务单，里面有 send/recv buffer、count、chunk、funcIndex 等。",
     "pccl 自研 host 时不必照抄 NCCL 函数名，但必须生成 PTX 能读懂的任务单布局。"),
    ("channelMask",
     "64-bit 位图，哪个 bit 为 1 就代表哪个 channel 参与本次 kernel。",
     "kernel 内部用 __popcll 把 blockIdx.x 映射到真实 channelId。"),
    ("P2P memory mapping",
     "一张 GPU 能直接访问另一张 GPU 暴露出来的 ring buffer。",
     "Ring LL 的 ld/st.volatile.global 依赖这个能力完成跨卡通信。"),
    ("dynamic shared memory / smem",
     "launch kernel 时额外申请的 shared memory 字节数。",
     "实测 RING_LL 也需要约 88416 B；不能简单以为 LL 就是 0。"),
]
for r in term_rows:
    ws4.append(list(r))

# ============================================================
# Sheet 5: 验证检查清单
# ============================================================
ws5 = wb.create_sheet("验证检查清单")
ws5.append(["优先级", "检查项", "通过标准", "对应依据"])
check_rows = [
    ("P0",
     "NCCL 公共 API 兼容",
     "能导出并链接 ncclAllReduce、ncclGetUniqueId、ncclCommInitRank/All，以及兼容的 datatype/reduce op 枚举。",
     "三层依赖表: ② 公共API"),
    ("P0",
     "AllReduce 参数语义正确",
     "count 按元素数而不是字节数解释；sendbuff/recvbuff 支持 in-place；stream 异步语义保持。",
     "allreduce-deep-dive.md §5"),
    ("P0",
     "device 结构体 ABI 兼容",
     "ncclDevComm、ncclDevChannel、ncclWork、ncclWorkElem 的字段偏移与 PTX 读取一致。",
     "④device层_ISA要求 / allreduce-deep-dive.md §11-12"),
    ("P0",
     "P2P ring buffer 可访问",
     "peer GPU buffer 能被本 GPU 通过 global memory load/store 访问，volatile 读写语义正确。",
     "④device层_ISA要求: ld/st.volatile.global"),
    ("P0",
     "kernel launch 参数一致",
     "传入 3 个参数 {devComm, channelMask, workHead}；grid.x=popcount(channelMask)；block/smem 与调度结果一致。",
     "allreduce-deep-dive.md §8"),
    ("P1",
     "RING+LL 路径可跑通",
     "至少复现实测 f16 all_reduce: grid=16、block=512、smem≈88416；以及 f32 barrier: grid=1、block=96。",
     "落地策略与约束: 实测依据"),
    ("P1",
     "barrier 复用 AllReduce",
     "torch barrier 最终能走 count=1 的 ncclAllReduce，不需要虚构 ncclBarrier。",
     "allreduce-deep-dive.md §barrier 变体"),
    ("P1",
     "PTX ISA 覆盖",
     "支持 __popcll、bar.sync 0、bar.sync %r,%n、ld/st.volatile.global 以及 shared/global 地址转换。",
     "④device层_ISA要求"),
    ("P2",
     "Generic kernel 边界清楚",
     "LL128/SIMPLE 不误用 RING_LL 特化 entry；需要时改走 ncclDevKernel_Generic。",
     "allreduce-deep-dive.md §12.3 注意事项"),
]
for r in check_rows:
    ws5.append(list(r))

# ============================================================
# 样式
# ============================================================
THIN = Side(style="thin", color="B0B0B0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEADER_FILL = PatternFill("solid", fgColor="305496")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
WRAP = Alignment(wrap_text=True, vertical="top", horizontal="left")

TIER_FILLS = {
    "①": PatternFill("solid", fgColor="DDEBF7"),
    "②": PatternFill("solid", fgColor="FFF2CC"),
    "③": PatternFill("solid", fgColor="FCE4D6"),
    "④": PatternFill("solid", fgColor="E2F0D9"),
}

PRIORITY_FILLS = {
    "P0": PatternFill("solid", fgColor="F4CCCC"),
    "P1": PatternFill("solid", fgColor="FCE5CD"),
    "P2": PatternFill("solid", fgColor="D9EAD3"),
}

def style_sheet(ws, widths):
    ws.sheet_view.showGridLines = False
    # header
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = BORDER
    # body
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = WRAP
            c.border = BORDER
        # tier color by first cell
        first = str(row[0].value or "")
        for key, fill in TIER_FILLS.items():
            if first.startswith(key):
                row[0].fill = fill
                break
        if first in PRIORITY_FILLS:
            row[0].fill = PRIORITY_FILLS[first]
            row[0].font = Font(bold=True)
    # column widths
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    # freeze header
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for i in range(2, ws.max_row + 1):
        ws.row_dimensions[i].height = 42

style_sheet(wb["先读导图"], [18, 76, 42])
style_sheet(ws,  [16, 22, 46, 26, 50, 46, 14])
style_sheet(ws2, [16, 48, 24, 52])
style_sheet(ws3, [26, 96])
style_sheet(ws4, [24, 58, 64])
style_sheet(ws5, [10, 34, 68, 42])

out = Path(__file__).with_name("ptx-compat-requirements.xlsx")
wb.save(out)
print("written:", out)
