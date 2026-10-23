"""KSR 与"同参数量稠密模型"的实测对照：decode 吞吐、显存/内存峰值、硬件门槛。

口径说明（这是关键，口径不对数字就没意义）：
  · "同等大小"按**总参数量**对齐。硬件门槛由总参数量决定，不是由激活参数量决定，
    所以只有按总参对齐才是对稠密模型有利的公平口径。
  · 两边都走同一套测量方法：同一个增量解码循环、同样的 prompt 长度、同样的 token 数、
    都带 KV/线性注意力缓存。稠密基线不是另写一个模型，而是把 KSR 的常识专家宽度解到
    刚好等于目标参数量、并且把所有层都设成全注意力——它就是标准稠密 transformer，
    只是跑在同一份代码上。
  · 每个模型单独一个进程跑，避免内存峰值互相污染；脚本自己只报 torch 侧的统计，
    进程峰值驻留由 peak_rss_bytes() 给出（CPU 口径），GPU 口径看 max_memory_*。
  · GPU 上必须 warmup + synchronize，否则测到的是内核启动和显存分配，不是吞吐。
  · batch>1 时各行的 prompt **不同**，这样路由才会选出不同专家、把多专家路径真的跑起来；
    全部用同一条 prompt 会对 KSR 特别有利（热集和专家选择都退化成单条）。
  · 计时只量解码循环（prefill 不计入），这才是"decode 吞吐"。

用法：  python tools/bench.py --config tools/configs/ksr_180m.json --side ksr
        python tools/bench.py --config tools/configs/ksr_180m.json --side dense
        python tools/bench.py --config tools/configs/ksr_180m.json --side ksr \
            --device cuda --dtype fp16 --batch 8 --verify
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ksr_calc                                    # noqa: E402
from ksr.config import KSRConfig                   # noqa: E402
from ksr.nets import CacheList, KSRModel           # noqa: E402


def dense_config_for(cfg: dict, target_params: int) -> dict:
    """把 cfg 改造成"同参数量的稠密模型"：所有层全注意力，FFN 宽度二分求解。

    常识专家就是稠密 FFN（始终激活），专家只留一个秩 8 的占位适配器（可忽略），
    这样基线就是标准稠密 transformer，而且用的是同一份前向代码。
    """
    d = copy.deepcopy(cfg)
    L = d["num_layers"]
    d["full_attn_layers"] = list(range(1, L + 1))       # 全部层做全注意力
    d["num_skill_experts"] = 1
    d["top_k_experts"] = 1
    d["expert_rank"] = 8
    d["corrector"] = dict(d.get("corrector", {}))
    d["corrector"]["enabled"] = False
    lo, hi = 8, 16 * d["hidden_size"]
    best = lo
    while lo <= hi:                                      # 二分：FFN 宽度对齐总参数量
        mid = (lo + hi) // 2
        d["reasoner_hidden"] = mid
        got = ksr_calc.analyze(copy.deepcopy(d))["params"]["total"]
        if got <= target_params:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    d["reasoner_hidden"] = best
    return d


def peak_rss_bytes() -> int:
    """本进程的峰值驻留内存。这就是"资源占用/硬件门槛"的实测口径：
    能不能跑得起来，取决于峰值驻留，不取决于参数量标签。"""
    try:
        import ctypes
        import ctypes.wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        m = PMC()
        m.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        # 必须显式声明返回类型：默认 restype 是 int32，会把伪句柄 -1 截断成无效句柄，
        # 于是调用"成功"但结构体全零——这就是峰值内存读成 0 的原因。
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC),
                                                ctypes.wintypes.DWORD]
        handle = k32.GetCurrentProcess()
        if not k32.K32GetProcessMemoryInfo(handle, ctypes.byref(m), m.cb):
            return 0
        return int(m.PeakWorkingSetSize)
    except Exception:                                   # noqa: BLE001
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def prompt_ids(vocab_size: int, prompt_len: int, row: int) -> list[int]:
    """第 row 条 prompt。各行必须不同，否则路由只会选同一个专家，多专家路径没被跑到。"""
    return [1] + [((row * 911 + 7) + i * 37 + 5) % (vocab_size - 10) + 10
                  for i in range(prompt_len)]


@torch.no_grad()
def decode_batch(model: KSRModel, ids: torch.Tensor, max_new: int,
                 collect: bool = False) -> list[int]:
    """增量解码：prefill 一次，然后逐 token 走 max_new 步。

    和 model.generate() 的区别只有两点，都是为了测吞吐：
      * 支持 batch>1（两种注意力的缓存都是按 batch 维排的，模型侧本来就支持）；
      * 贪心取样不做 .item()——每 token 同步一次会把 GPU 饿死，测出来的是同步延迟。
    这两条等价性由 --verify 守住：batch=1 时必须和 generate() 逐 token 一致。
    """
    model.eval()
    cache = CacheList(model.cfg.num_layers)
    logits, _ = model(ids, cache, 0)
    nxt = logits[:, -1, :]
    pos = ids.shape[1]
    out: list[int] = []
    for _ in range(max_new):
        nxt = nxt.argmax(-1, keepdim=True)             # [B,1]，留在设备上
        if collect:
            out.append(int(nxt[0, 0].item()))
        logits, _ = model(nxt, cache, pos)
        nxt = logits[:, -1, :]
        pos += 1
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--side", choices=["ksr", "dense"], required=True)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                    help="P40 是 sm_61，没有张量核也不支持 bf16，只有 fp16 这一条半精度路")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hot-experts", type=int, default=-1,
                    help="开启动态驻留，并给出热集合容量；-1 表示保持现状（全部常驻）。")
    ap.add_argument("--verify", action="store_true",
                    help="先证明 batch=1 的批量解码和 model.generate() 逐 token 一致，再谈数字")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"error": "cuda unavailable"}, ensure_ascii=False))
        return 2
    if args.dtype == "fp16" and args.device == "cpu":
        print(json.dumps({"error": "cpu 上 fp16 没有意义（没有内核加速，只会慢）"},
                         ensure_ascii=False))
        return 2

    raw = json.loads(Path(args.config).read_text("utf-8"))
    spec = ksr_calc.analyze(copy.deepcopy(raw))
    cfg = KSRConfig.from_json(args.config)

    if args.side == "dense":
        # 用解析式解出稠密 FFN 宽度，让总参数量与 KSR 对齐；再把所有层改成全注意力、
        # 专家缩成一个秩 8 的占位适配器（可忽略）。这样基线就是标准稠密 transformer，
        # 而且用的还是同一份前向代码——两边唯一的差别就是架构本身。
        d = dense_config_for(raw, spec["params"]["total"])
        cfg.full_attn_layers = list(range(1, cfg.num_layers + 1))
        cfg.num_skill_experts = 1
        cfg.top_k_experts = 1
        cfg.expert_rank = 8
        cfg.reasoner_hidden = d["reasoner_hidden"]
        cfg.corrector.enabled = False
    model = KSRModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device=device, dtype=(torch.float16 if args.dtype == "fp16" else torch.float32))
    model.eval()

    paged = args.hot_experts >= 0 and args.side == "ksr"
    if args.hot_experts >= 0 and not paged:
        print(json.dumps({"warning": "dense 基线只有一个专家，动态驻留无从谈起，已忽略该开关"},
                         ensure_ascii=False))
    if paged:
        # 动态驻留：只有热集合的专家栈在计算设备上，其余在 CPU，前向选中谁就搬谁。
        # 这是"显存只常驻常识专家 + 少数热专家"这条主张唯一能实测的口径——不开这个开关，
        # 测的其实是"全部专家常驻显存"的静态版本，和 README 写的设计不是一回事。
        hot = list(range(min(args.hot_experts, cfg.num_skill_experts)))
        # capacity 必须一起给：只设初始集合不设容量，前向会把其余专家调回来、一个都不换出
        model.experts.enable_paging(device, hot=hot, capacity=len(hot))
        if args.device == "cuda":
            torch.cuda.empty_cache()

    ids = torch.tensor([prompt_ids(cfg.vocab_size, args.prompt_len, r)
                        for r in range(args.batch)], device=device, dtype=torch.long)

    verify = None
    if args.verify:
        # 新写的解码循环必须先证明它和 generate() 走的是同一条语义，否则吞吐数字全是空的。
        single = prompt_ids(cfg.vocab_size, args.prompt_len, 0)
        gen_ids, _ = model.generate(list(single), max_new=min(args.tokens, 16))
        got = decode_batch(model, torch.tensor([single], device=device, dtype=torch.long),
                           min(args.tokens, 16), collect=True)
        verify = {"batch1_matches_generate": list(gen_ids) == got,
                  "generate": list(gen_ids), "batched": got}
        if not verify["batch1_matches_generate"]:
            print(json.dumps({"verify": verify, "fatal": "批量解码与 generate() 不一致，"
                                                         "后面的吞吐数字不可信"},
                             ensure_ascii=False))
            return 3

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 预热：把内核启动、显存分配、cuBLAS 选核算法都跑起来，否则第一个数字全是固定开销
    decode_batch(model, ids, 3)
    sync(device)
    loads0, evicts0 = model.experts.load_events, model.experts.evict_events
    t0 = time.perf_counter()
    decode_batch(model, ids, args.tokens)
    sync(device)
    dt = time.perf_counter() - t0

    out = {
        "side": args.side,
        "params": n_params,
        "activated_per_token": model.activated_params(),
        "reasoner_hidden": cfg.reasoner_hidden,
        "full_attn_layers": len(cfg.full_attn_layers),
        "linear_attn_layers": cfg.num_layers - len(cfg.full_attn_layers),
        "device": args.device, "dtype": args.dtype, "batch": args.batch,
        "tokens": args.tokens, "seconds": dt,
        "tok_per_s": args.batch * args.tokens / dt,          # 批量时是总吞吐
        "tok_per_s_per_seq": args.tokens / dt,               # 单序列延迟口径
        "peak_rss_gb": round(peak_rss_bytes() / 1e9, 2),
        "prompt_len": args.prompt_len,
    }
    if args.device == "cuda":
        out["peak_gpu_alloc_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
        out["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1e9, 3)
        out["gpu_name"] = torch.cuda.get_device_name(device)
    if paged:
        out["hot_experts"] = len(model.experts.hot)
        out["expert_resident_gb"] = round(model.experts.resident_bytes / 1e9, 3)
        # 只统计计时窗口里的搬运：预热阶段的换入换出是冷启动，不该算进稳态成本
        out["expert_loads_per_step"] = round((model.experts.load_events - loads0) / args.tokens, 2)
        out["expert_evicts_per_step"] = round((model.experts.evict_events - evicts0) / args.tokens, 2)
    if verify is not None:
        out["verify"] = verify
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
