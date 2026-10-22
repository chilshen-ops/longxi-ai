#!/usr/bin/env python3
"""KSR-MoE 规格计算器 —— 所有规格数字的唯一来源。

用法：
    python tools/ksr_calc.py tools/configs/ksr7b.json            # 输出 markdown 到 stdout
    python tools/ksr_calc.py tools/configs/ksr7b.json --write docs/NUMBERS.md
    python tools/ksr_calc.py --selfcheck                          # 与独立手算值核对（±1%）
    python tools/ksr_calc.py --all                                # 逐个配置输出对比表

设计原则：
  * 只依赖标准库，任何机器都能跑；
  * 每个公式都写出来源，不做"估一个大概"；
  * 断言失败即以非零码退出，规格文档因此不可能与代码脱节。

公式约定（全部按"每 token"口径，除注明外）：
  线性层参数   = in_features * out_features（bias 忽略，占比 <0.1%）
  矩阵乘 FLOPs = 2 * 参数量（一次乘加记 2 FLOPs）
  全注意力额外 = 2 * (heads*head_dim) * n^2 / 序列（prefill，因果掩码取 1/2）
                 4 * (heads*head_dim) * n（decode 单 token，读满 n 个历史 token）
  线性注意力额外 = 4 * heads * dk * dv（delta rule 的写状态 + 读状态，近似式）
  KV Cache     = 2(K,V) * kv_groups * head_dim * ctx * bytes，仅全注意力层持有
  线性状态     = heads * dk * dv * bytes，与序列长度无关（这是混合注意力的主要收益）
  权重大小     = 参数量 * bytes_per_param(量化档)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ── 量化档：每参数字节数（含分组 scale/zero-point 的摊薄开销）───────────────────
# int4g32: 4bit 权重 + 每 32 个参数一份 fp16 scale → 0.5 + 2/32
# int8g64: 8bit 权重 + 每 64 个参数一份 int8 scale → 1 + 1/64
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8_e4m3": 1.0,
    "int8": 1.0,
    "int8g64": 1.0 + 1.0 / 64.0,
    "int4": 0.5,
    "int4g32": 0.5 + 2.0 / 32.0,
}


def nbytes(dtype: str) -> float:
    try:
        return BYTES_PER_PARAM[dtype]
    except KeyError:
        raise SystemExit(f"未知量化档: {dtype}；可选: {sorted(BYTES_PER_PARAM)}")


# ── 参数量公式 ────────────────────────────────────────────────────────────────
def attn_proj_params(h: int, heads: int, kv_groups: int, head_dim: int) -> int:
    """Q/K/V/O 投影。kv_groups == heads 时退化为 MHA。"""
    q = h * heads * head_dim
    k = h * kv_groups * head_dim
    v = h * kv_groups * head_dim
    o = heads * head_dim * h
    return q + k + v + o


def ffn_params(kind: str, h: int, f: int) -> int:
    """FFN。swiglu = gate/up/down 三矩阵（Llama 风格）；silu2 = up/down 两矩阵。"""
    if kind == "swiglu":
        return 3 * h * f
    if kind == "silu2":
        return 2 * h * f
    raise SystemExit(f"未知 FFN 类型: {kind}（应为 swiglu | silu2）")


def lowrank_params(h: int, rank: int, kind: str = "swiglu") -> int:
    """低秩旁路专家：h -> rank -> h。这就是"技能专家"的实现形态。"""
    return ffn_params(kind, h, rank)


def router_params(h: int, hidden: int, num_experts: int) -> int:
    """h→hidden→(专家 logits) + 一个 need_knowledge 头（含 bias）。与 ksr/nets.py 逐项对应。"""
    return h * hidden + hidden * num_experts + hidden + 1


def corrector_params(h: int, hidden: int, classes: int, with_correction_proj: bool) -> int:
    """序列级校验器：[主输出; 常识输出] -> hidden -> classes，可选一个纠正投影。"""
    p = (2 * h) * hidden + hidden * classes
    if with_correction_proj:
        p += h * h
    return p


# ── 分析 ──────────────────────────────────────────────────────────────────────
def analyze(cfg: dict) -> dict:
    h = cfg["hidden_size"]
    L = cfg["num_layers"]
    heads = cfg["num_attention_heads"]
    kg = cfg["num_kv_groups"]
    hd = cfg.get("head_dim") or (h // heads)
    V = cfg["vocab_size"]
    ctx = cfg["context_len"]
    q = cfg["quant"]
    lin = cfg["linear_attn"]
    corr = cfg.get("corrector", {"enabled": False})

    full_layers = cfg.get("full_attn_layers") or list(range(1, L + 1))
    n_full = len(full_layers)
    n_lin = L - n_full
    assert n_full >= 0 and n_lin >= 0, "full_attn_layers 越界"

    # 线性注意力层的投影预算：默认与 GQA 全注意力层同预算（共享 K/V 投影的 DeltaNet）
    lin_kg = lin.get("num_kv_groups", kg)
    lin_dk = lin.get("head_dim_k", hd)
    lin_dv = lin.get("head_dim_v", hd)
    lin_heads = lin.get("num_v_heads", heads)
    # 衰减门 + 写入门（含 bias），与 ksr/nets.py 的 GatedDeltaNet 逐项对应
    lin_gates = 2 * (h * lin_heads + lin_heads)

    # --- 参数分解 ---
    p_attn_full = attn_proj_params(h, heads, kg, hd) * n_full
    p_attn_lin = (attn_proj_params(h, lin_heads, lin_kg, lin_dv) + lin_gates) * n_lin
    p_attn = p_attn_full + p_attn_lin

    ffn_kind = cfg["ffn_type"]
    p_reasoner = ffn_params(ffn_kind, h, cfg["reasoner_hidden"]) * L

    n_exp = cfg["num_skill_experts"]
    rank = cfg["expert_rank"]
    p_expert_unit = lowrank_params(h, rank, cfg.get("expert_ffn_type", ffn_kind))
    p_experts_pool = p_expert_unit * n_exp * L
    top_k = cfg["top_k_experts"]
    p_experts_active = p_expert_unit * top_k * L

    p_router = router_params(h, cfg["router_hidden"], n_exp) * L

    p_corrector = 0
    corrector_active = False
    if corr.get("enabled"):
        p_corrector = corrector_params(
            h, corr["hidden"], corr.get("num_classes", 3), corr.get("with_correction_proj", True)
        )
        copies = L if corr.get("placement") == "per_layer" else 1
        p_corrector *= copies
        corrector_active = corr.get("always_on", True)

    p_embed = V * h
    p_head = 0 if cfg.get("tied_embeddings", True) else V * h
    p_norms = h * (3 * L + 1)  # 每层 attn/reasoner/expert 三个 RMSNorm + 末尾一个（与 ksr/nets.py 一致）

    total = (p_attn + p_reasoner + p_experts_pool + p_router + p_corrector
             + p_embed + p_head + p_norms)

    # 输出投影的计算量与权重是否绑定无关：绑定省的是权重字节，不是每 token 的矩阵乘。
    p_head_compute = V * h
    activated = (p_attn + p_reasoner + p_router + p_experts_active + p_head_compute + p_norms)
    if corrector_active:
        activated += p_corrector

    # --- FLOPs ---
    d_full = heads * hd
    d_lin = lin_heads * lin_dv
    proj_flops_tok = 2 * activated  # 含 lm_head；embedding 查表不计 FLOPs

    # prefill：全注意力层 O(n^2) 项（因果掩码取 1/2）
    attn_extra_prefill = n_full * 2 * d_full * ctx * ctx
    lin_extra_prefill = n_lin * 4 * lin_heads * lin_dk * lin_dv * ctx
    prefill_flops = proj_flops_tok * ctx + attn_extra_prefill + lin_extra_prefill

    # decode：单 token，读满 ctx 个历史
    attn_extra_decode = n_full * 4 * d_full * ctx
    lin_extra_decode = n_lin * 4 * lin_heads * lin_dk * lin_dv
    decode_flops = proj_flops_tok + attn_extra_decode + lin_extra_decode

    # 纯 Transformer 对照（同层数、全注意力）的注意力项，用于验证"注意力省到 30%"的说法
    attn_extra_dense = L * 2 * d_full * ctx * ctx
    attn_ratio = (attn_extra_prefill / attn_extra_dense) if attn_extra_dense else 1.0

    # --- 显存 ---
    w_attn = p_attn * nbytes(q["attention"])
    w_reasoner = p_reasoner * nbytes(q["reasoner"])
    w_router = p_router * nbytes(q["router"])
    w_experts_all = p_experts_pool * nbytes(q["experts"])
    hot = min(cfg.get("hot_experts", n_exp), n_exp)
    w_experts_hot = p_expert_unit * hot * L * nbytes(q["experts"])
    w_experts_cold = w_experts_all - w_experts_hot
    w_embed = (p_embed + p_head) * nbytes(q["embedding"])
    w_corrector = p_corrector * nbytes(q.get("corrector", "bf16"))
    w_total = w_attn + w_reasoner + w_router + w_experts_all + w_embed + w_corrector

    kv_bytes = n_full * 2 * kg * hd * ctx * nbytes(q.get("kv_cache", "bf16"))
    state_bytes = n_lin * lin_heads * lin_dk * lin_dv * nbytes(q.get("kv_cache", "bf16"))
    overhead = cfg.get("runtime_overhead_bytes", 0)

    resident = w_attn + w_reasoner + w_router + w_experts_hot + w_embed + w_corrector + kv_bytes + state_bytes + overhead

    # --- 冷专家搬运 ---
    bw = cfg.get("link", {}).get("bandwidth_gbps", 50.0) * 1e9
    cold_stack_bytes = p_expert_unit * L * nbytes(q["experts"])
    load_ms = cold_stack_bytes / bw * 1000.0

    return {
        "name": cfg.get("name", "unnamed"),
        "cfg": cfg,
        "params": {
            "attention_full": p_attn_full,
            "attention_linear": p_attn_lin,
            "attention": p_attn,
            "reasoner": p_reasoner,
            "experts_pool": p_experts_pool,
            "experts_unit": p_expert_unit,
            "router": p_router,
            "corrector": p_corrector,
            "embedding": p_embed,
            "lm_head": p_head,
            "norms": p_norms,
            "total": total,
            "activated_per_token": activated,
        },
        "flops": {
            "proj_per_token": proj_flops_tok,
            "prefill_total": prefill_flops,
            "decode_per_token": decode_flops,
            "attn_extra_prefill": attn_extra_prefill,
            "attn_extra_dense": attn_extra_dense,
            "attn_ratio_vs_dense_same_depth": attn_ratio,
        },
        "vram": {
            "attention": w_attn,
            "reasoner": w_reasoner,
            "router": w_router,
            "experts_hot": w_experts_hot,
            "experts_cold": w_experts_cold,
            "experts_all": w_experts_all,
            "embedding": w_embed,
            "corrector": w_corrector,
            "kv_cache": kv_bytes,
            "linear_states": state_bytes,
            "overhead": overhead,
            "weights_total": w_total,
            "resident": resident,
        },
        "offload": {
            "hot_experts": hot,
            "cold_experts": n_exp - hot,
            "cold_stack_bytes": cold_stack_bytes,
            "load_ms": load_ms,
            "bandwidth_gbps": bw / 1e9,
        },
        "layers": {"total": L, "full_attn": n_full, "linear_attn": n_lin},
        "context_len": ctx,
    }


# ── 稠密基线（固定，写死在代码里，避免"基线随手调"）────────────────────────────
def dense_baseline(kind: str, quant: dict) -> dict:
    """kind: 'llama7b'（32 层 MHA，真 7B）| 'same_base'（24 层 GQA，与 KSR 同底座）"""
    if kind == "llama7b":
        L, kg = 32, 32  # MHA
    elif kind == "same_base":
        L, kg = 24, 4
    else:
        raise SystemExit(f"未知基线: {kind}")
    cfg = {
        "name": f"dense-{kind}",
        "hidden_size": 4096,
        "num_layers": L,
        "num_attention_heads": 32,
        "num_kv_groups": kg,
        "vocab_size": 131_072,
        "context_len": 32_768,
        "full_attn_layers": list(range(1, L + 1)),
        "linear_attn": {"num_kv_groups": kg},
        "ffn_type": "swiglu",
        "reasoner_hidden": 11_008,
        "num_skill_experts": 0,
        "expert_rank": 0,
        "top_k_experts": 0,
        "router_hidden": 0,
        "corrector": {"enabled": False},
        "tied_embeddings": True,
        "quant": quant,
        "hot_experts": 0,
        "link": {"bandwidth_gbps": 50.0},
    }
    return analyze(cfg)


# ── 输出 ──────────────────────────────────────────────────────────────────────
def b(x: float) -> str:
    """字节 → GB（十进制，与显存厂商标称一致）。"""
    return f"{x / 1e9:.2f} GB"


def bB(x: float) -> str:
    return f"{x / 1e9:.2f}B"


def gf(x: float) -> str:
    return f"{x / 1e9:.1f} GFLOPs"


def render(cfg: dict, res: dict) -> str:
    p, f, v, o = res["params"], res["flops"], res["vram"], res["offload"]
    ctx = res["context_len"]
    L = res["layers"]
    out = []
    w = out.append
    w(f"## {res['name']} — 规格核算\n")
    w(f"配置来源：`{cfg.get('_path','(inline)')}`；上下文 {ctx} tokens；"
      f"{L['total']} 层（{L['full_attn']} 全注意力 + {L['linear_attn']} 线性注意力）。\n")

    w("### 1. 参数量\n")
    w("| 组件 | 参数量 | 每层 | 说明 |")
    w("|---|---|---|---|")
    w(f"| Attention（全注意力 {L['full_attn']} 层） | {bB(p['attention_full'])} | "
      f"{bB(p['attention_full']/max(L['full_attn'],1))} | GQA，K/V 共享投影 |")
    w(f"| Attention（线性 {L['linear_attn']} 层） | {bB(p['attention_linear'])} | "
      f"{bB(p['attention_linear']/max(L['linear_attn'],1))} | 投影预算与 GQA 层相同 |")
    w(f"| 常识专家（常驻） | {bB(p['reasoner'])} | {bB(p['reasoner']/L['total'])} | 每层都激活 |")
    w(f"| 技能专家池（{cfg['num_skill_experts']} 个） | {bB(p['experts_pool'])} | "
      f"{bB(p['experts_pool']/L['total'])} | 单专家单层 {bB(p['experts_unit'])} |")
    w(f"| 路由 | {bB(p['router'])} | {bB(p['router']/L['total'])} | 轻量门控 |")
    w(f"| 纠错器 | {bB(p['corrector'])} | — | "
      f"{cfg.get('corrector',{}).get('placement','n/a')} |")
    w(f"| Embedding + LM Head | {bB(p['embedding']+p['lm_head'])} | — | "
      f"{'绑定' if cfg.get('tied_embeddings', True) else '不绑定'} |")
    w(f"| **总参数** | **{bB(p['total'])}** | | |")
    w(f"| **每 token 激活** | **{bB(p['activated_per_token'])}** | | "
      f"Top-{cfg['top_k_experts']} 专家 + 常识专家 + 注意力 + LM Head |")
    w("")

    w("### 2. 算力\n")
    w("| 指标 | 数值 |")
    w("|---|---|")
    w(f"| 投影矩阵乘 / token | {gf(f['proj_per_token'])} |")
    w(f"| Prefill（{ctx} tokens，整段） | {gf(f['prefill_total'])} |")
    w(f"| Decode / token（{ctx} 上下文） | {gf(f['decode_per_token'])} |")
    w(f"| 注意力额外算力 / 同深度稠密 | {f['attn_ratio_vs_dense_same_depth']*100:.1f}% |")
    w("")

    w("### 3. 显存（推理常驻）\n")
    w("| 组件 | 占用 |")
    w("|---|---|")
    w(f"| Attention 权重 | {b(v['attention'])} |")
    w(f"| 常识专家权重 | {b(v['reasoner'])} |")
    w(f"| 路由权重 | {b(v['router'])} |")
    w(f"| 技能专家（常驻 {o['hot_experts']} 个） | {b(v['experts_hot'])} |")
    w(f"| 技能专家（冷 {o['cold_experts']} 个，不占显存） | {b(v['experts_cold'])} |")
    w(f"| Embedding + LM Head | {b(v['embedding'])} |")
    w(f"| 纠错器 | {b(v['corrector'])} |")
    w(f"| KV Cache（{ctx}，仅 {L['full_attn']} 层） | {b(v['kv_cache'])} |")
    w(f"| 线性注意力状态（{L['linear_attn']} 层，与长度无关） | {b(v['linear_states'])} |")
    if v["overhead"]:
        w(f"| 运行时开销（激活值/碎片，估算） | {b(v['overhead'])} |")
    w(f"| **常驻合计** | **{b(v['resident'])}** |")
    w(f"| 全部权重合计（含冷专家） | {b(v['weights_total'])} |")
    w("")

    w("### 4. 冷专家搬运\n")
    w(f"单个专家栈 = {L['total']} 层适配器 = {b(o['cold_stack_bytes'])}；"
      f"按 {o['bandwidth_gbps']:.0f} GB/s 单向带宽，加载耗时 **{o['load_ms']:.1f} ms**。\n")

    # 与基线对比
    q = cfg["quant"]
    for base_kind, base_label in (("llama7b", "稠密 7B（32 层 MHA）"),
                                  ("same_base", f"同底座稠密（{L['total']} 层 GQA）")):
        base = dense_baseline(base_kind, q)
        bp, bf_, bv = base["params"], base["flops"], base["vram"]
        w(f"### 5. 对比：{base_label}\n")
        w("| 指标 | 本架构 | 基线 | 比值 |")
        w("|---|---|---|---|")
        w(f"| 总参数 | {bB(p['total'])} | {bB(bp['total'])} | {p['total']/bp['total']*100:.0f}% |")
        w(f"| 每 token 激活参数 | {bB(p['activated_per_token'])} | {bB(bp['activated_per_token'])} | "
          f"{p['activated_per_token']/bp['activated_per_token']*100:.0f}% |")
        w(f"| Prefill FLOPs / token | {gf(f['prefill_total']/ctx)} | {gf(bf_['prefill_total']/ctx)} | "
          f"{f['prefill_total']/bf_['prefill_total']*100:.0f}% |")
        w(f"| Decode FLOPs / token | {gf(f['decode_per_token'])} | {gf(bf_['decode_per_token'])} | "
          f"{f['decode_per_token']/bf_['decode_per_token']*100:.0f}% |")
        w(f"| 权重显存 | {b(v['weights_total'] - v['experts_cold'])} | {b(bv['weights_total'])} | "
          f"{(v['weights_total']-v['experts_cold'])/bv['weights_total']*100:.0f}% |")
        w(f"| KV Cache（{ctx}） | {b(v['kv_cache']+v['linear_states'])} | {b(bv['kv_cache'])} | "
          f"{(v['kv_cache']+v['linear_states'])/bv['kv_cache']*100:.0f}% |")
        w(f"| 常驻显存合计 | {b(v['resident'])} | {b(bv['resident'])} | "
          f"{v['resident']/bv['resident']*100:.0f}% |")
        w("")
    return "\n".join(out)


# ── 自检：与独立手算值核对（±1%）──────────────────────────────────────────────
def selfcheck() -> int:
    h, L, heads, hd, kg = 4096, 24, 32, 128, 4
    # 独立手算（不经过 analyze()）
    hand_attn_layer = 4096*4096 + 2*(4096*512) + 4096*4096          # 37,748,736
    assert attn_proj_params(h, heads, kg, hd) == hand_attn_layer, attn_proj_params(h, heads, kg, hd)
    assert ffn_params("swiglu", h, 1024) == 3 * 4096 * 1024 == 12_582_912
    assert lowrank_params(h, 512, "swiglu") == 6_291_456
    assert attn_proj_params(h, heads, heads, hd) == 4 * 4096 * 4096 == 67_108_864  # MHA

    cfg = json.loads((Path(__file__).resolve().parent / "configs/ksr7b.json").read_text("utf-8"))
    r = analyze(cfg)
    p = r["params"]
    # 逐项手算期望值
    exp = {
        # 6 个全注意力层 + 18 个线性层（线性层多两个门控投影）
        "attention": 6 * 37_748_736 + 18 * (37_748_736 + 2 * (4096 * 32 + 32)),
        "reasoner": 12_582_912 * 24,                       # 301,989,888
        "experts_pool": 6_291_456 * 16 * 24,               # 2,415,919,104
        "experts_active_hand": 6_291_456 * 24,             # 150,994,944
        "embedding": 131_072 * 4096,                       # 536,870,912
    }
    for k, want in exp.items():
        got = p["experts_active_per_token_hand"] if False else p.get(k, p.get("activated_per_token"))
        if k == "experts_active_hand":
            got = p["experts_unit"] * 24
        assert abs(got - want) / want < 1e-6, (k, got, want)

    # 稠密 7B 基线必须真的是 7B 量级（这是"7B"标签的判据）
    base = dense_baseline("llama7b", cfg["quant"])
    t = base["params"]["total"]
    assert 6.9e9 < t < 7.2e9, f"llama7b 基线应为 ~7.0B，实得 {t/1e9:.2f}B"
    sb = dense_baseline("same_base", cfg["quant"])
    assert 4.5e9 < sb["params"]["total"] < 4.9e9, f"同底座稠密应为 ~4.7B，实得 {sb['params']['total']/1e9:.2f}B"

    # 注意力占比：6/24 层全注意力 → 应为 6/24 = 25%
    assert abs(r["flops"]["attn_ratio_vs_dense_same_depth"] - 6/24) < 1e-9, r["flops"]["attn_ratio_vs_dense_same_depth"]

    # 线性状态与长度无关
    v1 = analyze({**cfg, "context_len": 1024})["vram"]["linear_states"]
    v2 = analyze({**cfg, "context_len": 131072})["vram"]["linear_states"]
    assert v1 == v2
    return 0


# ── 断言：规格必须守住的边界 ───────────────────────────────────────────────────
def check_limits(cfg: dict, res: dict) -> list[str]:
    lim = cfg.get("limits", {})
    fails = []
    p, v = res["params"], res["vram"]
    tests = [
        ("总参数", p["total"], lim.get("max_total_params")),
        ("每 token 激活参数", p["activated_per_token"], lim.get("max_activated_params")),
        ("常驻显存", v["resident"], (lim.get("max_resident_gib") or 0) * 1e9 or None),
        ("专家池参数", p["experts_pool"], lim.get("max_expert_pool_params")),
    ]
    for name, got, want in tests:
        if want is None:
            continue
        if got > want:
            fails.append(f"{name} {got/1e9:.2f}B/GB 超出上限 {want/1e9:.2f}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--write", help="把 markdown 写到该路径")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--all", action="store_true", help="输出 configs/ 下所有配置的对比")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    cfg_dir = Path(__file__).resolve().parent / "configs"
    if args.all:
        paths = sorted(cfg_dir.glob("*.json"))
    elif args.config:
        paths = [Path(args.config)]
    else:
        paths = [cfg_dir / "ksr7b.json"]

    rc = 0
    blocks = []
    for path in paths:
        cfg = json.loads(path.read_text("utf-8"))
        cfg["_path"] = str(path.relative_to(Path(__file__).resolve().parents[1]))
        res = analyze(cfg)
        fails = check_limits(cfg, res)
        if args.json:
            blocks.append(json.dumps({"config": cfg["_path"], "result": res, "limit_failures": fails},
                                     ensure_ascii=False, indent=2))
        else:
            md = render(cfg, res)
            if fails:
                md += "\n### ❌ 规格越界\n" + "\n".join(f"- {x}" for x in fails) + "\n"
                rc = 1
            else:
                md += "\n### ✅ 规格边界全部满足\n"
            blocks.append(md)
    text = "\n\n---\n\n".join(blocks)
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
