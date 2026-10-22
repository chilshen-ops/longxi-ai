"""KSR 与"同参数量稠密模型"的实测对照：decode 吞吐、内存峰值、硬件门槛。

口径说明（这是关键，口径不对数字就没意义）：
  · "同等大小"按**总参数量**对齐。硬件门槛由总参数量决定，不是由激活参数量决定，
    所以只有按总参对齐才是对稠密模型有利的公平口径。
  · 两边都走同一套测量方法：同一个 generate() 解码循环、同样的 batch=1、同样的
    prompt 长度、同样的 token 数、都带 KV/线性注意力缓存。稠密基线不是另写一个模型，
    而是把 KSR 的常识专家宽度解到刚好等于目标参数量、并且把所有层都设成全注意力
    ——它就是标准稠密 transformer，只是跑在同一份代码上。
  · 每个模型单独一个进程跑，避免内存峰值互相污染；峰值内存由调用方用
    PeakWorkingSet64 读，脚本自己只报 torch 侧的统计。

用法：  python tools/bench.py --config tools/configs/ksr_180m.json --side ksr
        python tools/bench.py --config tools/configs/ksr_180m.json --side dense
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
from ksr.nets import KSRModel                      # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--side", choices=["ksr", "dense"], required=True)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    raw = json.loads(Path(args.config).read_text("utf-8"))
    spec = ksr_calc.analyze(copy.deepcopy(raw))
    cfg = KSRConfig.from_json(args.config)

    if args.side == "dense":
        # 用解析式解出稠密 FFN 宽度，让总参数量与 KSR 对齐；再把所有层改成全注意力、
        # 专家缩成一个秩 8 的占位适配器（可忽略）。这样基线就是标准稠密 transformer，
        # 而且跑在同一份前向代码上——两边唯一的差别就是架构本身。
        d = dense_config_for(raw, spec["params"]["total"])
        cfg.full_attn_layers = list(range(1, cfg.num_layers + 1))
        cfg.num_skill_experts = 1
        cfg.top_k_experts = 1
        cfg.expert_rank = 8
        cfg.reasoner_hidden = d["reasoner_hidden"]
        cfg.corrector.enabled = False
    model = KSRModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    ids = [1] + [(i * 37 + 5) % (cfg.vocab_size - 10) + 10 for i in range(args.prompt_len)]
    with torch.no_grad():                       # 预热，把内存分配和内核都跑起来
        model.generate(list(ids), max_new=3)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(list(ids), max_new=args.tokens)
    dt = time.perf_counter() - t0

    print(json.dumps({
        "side": args.side,
        "params": n_params,
        "activated_per_token": model.activated_params(),
        "reasoner_hidden": cfg.reasoner_hidden,
        "full_attn_layers": len(cfg.full_attn_layers),
        "linear_attn_layers": cfg.num_layers - len(cfg.full_attn_layers),
        "tokens": args.tokens, "seconds": dt,
        "tok_per_s": args.tokens / dt,
        "peak_rss_gb": round(peak_rss_bytes() / 1e9, 2),
        "prompt_len": args.prompt_len,
        "device": "cpu" if model.embed.weight.device.type == "cpu" else "cuda",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
