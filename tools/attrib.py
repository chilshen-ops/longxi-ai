"""知识归属探针：一条知识到底在专家里、在知识库里、在共享通路里，还是哪儿都不在。

这个工具存在的原因是一条实测教训：把 16 个技能专家的混合权重**全部清零**，三组题的正确率
一个都不掉（56%→56%、75%→75%、8%→8%）。也就是说"专家承载领域知识"这件事在这个 checkpoint
里根本没发生，而在此之前它只是被默认假设着。归属不能靠声明，只能靠测。

四个探针，每个都对应一条归属规则（规则本身写在 docs/KNOWLEDGE.md）：

  gap_experts   开专家 − 全部专家清零            专家里到底有没有知识
  gap_shared_ffn 开全部 − 常识 FFN 清零          共享通路里有没有
  resident_only  两个 FFN 都关掉只剩注意力       知识是不是住在必须常驻、MoE 动不了的地方
  gap_kb        检索开 − 检索关（事实题）        事实到底在不在知识库里
  upper_bound   答案手抄进题面 − 检索开          缺的是"取到"还是"会用"
  conflicts     无检索时答案与 KB 不一致          同一条知识是不是有两个权威来源
  reliance      先验答错时是否改口采纳 KB 的值    已验证的知识能不能压过先验

外加一个判定：自然路由 vs 不用专家。前者更低就说明路由在帮倒忙。

用法：
    python tools/attrib.py --work work-scaled
    python tools/attrib.py --work work-scaled --json work-scaled/attrib.json

配置从检查点里记的 cfg_path 自动解析（--config 可覆盖）。**KSR_INTENTS 必须和训练时一致**，
否则重建出的留出集不是训练时那一个，测出来的准确率不可比。
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

# --work 必须在 import ksr.experiment 之前定下来：WORK 是模块级读的
if "--work" in sys.argv:
    os.environ["KSR_WORK"] = sys.argv[sys.argv.index("--work") + 1]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from ksr.config import KSRConfig  # noqa: E402
from ksr.nets import KSRModel  # noqa: E402
import ksr.experiment as E  # noqa: E402

CFG_DEFAULT = ROOT / "tools" / "configs" / "ksr_tiny.json"


def resolve_cfg_path(state: dict) -> str:
    """用哪个配置重建模型，优先级：--config > 检查点里记的配置 > KSR_CONFIG > 默认。

    踩过一次：探针硬编码 ksr_tiny_scaled.json，而 `python -m ksr.experiment` 默认用
    ksr_tiny.json，加载时直接 size mismatch。检查点本身不存配置，所以这里把 cfg_path 也
    存进 model.pt（见 experiment.py 保存处），探针就能自己找到正确的配置。
    """
    if "--config" in sys.argv:
        return sys.argv[sys.argv.index("--config") + 1]
    return (state.get("cfg_path") or os.environ.get("KSR_CONFIG") or str(CFG_DEFAULT))


def _zero_experts_handles(model):
    """把每个路由给的混合权重清零：技能专家全部不参与，只剩常识通路 + 检索上下文。"""

    def kill(_m, _i, out):
        out = dict(out)
        out["expert_weights"] = torch.zeros_like(out["expert_weights"])
        return out

    return [blk.router.register_forward_hook(kill) for blk in model.blocks]


def _zero_output_handles(model, attr):
    """把某个子模块（常识 FFN）的输出清零。

    和清零专家用的是同一种探针：消融意义上的归属，不是机制意义上的——清零是分布外干预，
    掉多少只能说"模型多依赖它"。但三个存储位置必须用同一种探针测，否则结论不可比。
    """

    def kill(_m, _i, out):
        return torch.zeros_like(out)

    return [getattr(blk, attr).register_forward_hook(kill) for blk in model.blocks]


def main() -> int:
    torch.set_num_threads(int(os.environ.get("KSR_THREADS", "2")))
    work = Path(os.environ["KSR_WORK"])
    ckpt = work / "model.pt"
    if not ckpt.exists():
        print(f"找不到检查点 {ckpt}（先跑 python -m ksr.experiment）")
        return 2

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_path = resolve_cfg_path(state)
    print(f"用配置 {cfg_path}")
    cfg = KSRConfig.from_json(cfg_path)
    corpus = E.build_corpus()
    tok = E.build_tokenizer(corpus)
    E._TOK = tok
    cfg.vocab_size = tok.vocab_size
    model = KSRModel(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    embed = E.make_embed(model)

    # ── 1. 三个存储位置，三次清零：知识到底在谁那里 ─────────────────────────
    # 清零是分布外干预，掉多少只能说"模型多依赖那个位置"，不是机制级归属；但三个位置必须
    # 用同一种探针测，结论才可比。留出集只有 32 题时 95% 置信半宽是 ±17.6pp，
    # 所以任何几个百分点的差都要连着 ± 号一起读，只有"0 题变化"是确定的。
    skill = corpus["eval_skill"]
    n_skill = max(len(skill), 1)

    def ci95(acc: float) -> float:
        p = (acc + 0.5) / 2                       # add-1/2 平滑，避免 0%/100% 时区间退化成 0
        return 2 * (p * (1 - p) / n_skill) ** 0.5

    all_on = E.eval_skill(model, tok, skill, with_correction=False)["acc"]
    h_exp = _zero_experts_handles(model)
    experts_off = E.eval_skill(model, tok, skill, with_correction=False)["acc"]
    for h in h_exp:
        h.remove()
    h_ffn = _zero_output_handles(model, "reasoner")
    ffn_off = E.eval_skill(model, tok, skill, with_correction=False)["acc"]
    h_exp2 = _zero_experts_handles(model)          # 在 FFN 已关的基础上再叠上专家
    resident_only = E.eval_skill(model, tok, skill, with_correction=False)["acc"]
    for h in list(h_ffn) + list(h_exp2):
        h.remove()
    gap_experts = all_on - experts_off
    gap_shared_ffn = all_on - ffn_off

    # ── 2/3. 事实在不在 KB 里；缺的是取到还是会用 ──────────────────────────
    kb = E.build_kb(model, tok, embed, corpus, cfg)
    kb_on = E.eval_fact(model, tok, kb, corpus["eval_fact"], embed, True, cfg)
    kb_off = E.eval_fact(model, tok, kb, corpus["eval_fact_noctx"], embed, False, cfg)
    upper_items = [(E.fact_line(p, svc) + "\n" + E.fact_query(p), svc, "fact")
                   for p, svc in ((p, corpus["fact_table"][p]) for p in corpus["ports_held"])]
    upper = E.eval_fact(model, tok, kb, upper_items, embed, False, cfg)
    gap_kb = kb_on["acc"] - kb_off["acc"]

    # ── 4. 同一条知识有没有两个权威来源 ────────────────────────────────────
    # 无检索也能答对 = 这条事实同时在权重里；若它还与 KB 不一致，更新 KB 就不会生效。
    dup = conflict = 0
    for prompt, gold, _dom in corpus["eval_fact_noctx"]:
        pred, _ = E.answer_of(model, tok, prompt)
        if pred == gold:
            dup += 1
        elif pred in set(corpus["fact_table"].values()):
            conflict += 1

    # ── 5. 证据优先率：模型自己的先验会不会让位给经过审核的知识条目 ──────────
    # 上面那个 conflict 计数说明剩下的准确率缺口不在检索也不在呈现（那两个 gap 都是 0），
    # 而在**先验压过证据**。这条直接测它：先给无上下文的答案（模型的先验），再给带检索的
    # 答案，看它是否改口采纳 KB 的值。"常识专家把关"如果意味着"以已验证的知识为准"，
    # 这个数就必须接近 1；否则 KB 更新和纠错都白做——模型根本不认账。
    ports = corpus["ports_held"]
    prior_wrong = followed = hurt = 0
    for p in ports:
        gold = corpus["fact_table"][p]
        q = E.fact_query(p)
        prior, _ = E.answer_of(model, tok, q)
        segs = E.retrieve_for(model, tok, kb, embed, q, cfg)
        with_ev, _ = E.answer_of(model, tok, E.render_context(segs) + q)
        if prior != gold:
            prior_wrong += 1
            followed += int(with_ev == gold)
        if prior == gold and with_ev != gold:
            hurt += 1                      # 检索反而把对的改错：干扰项的代价
    reliance = followed / prior_wrong if prior_wrong else 1.0

    out = {
        "work": str(work),
        "eval_skill_n": len(skill),
        "ci95_halfwidth": round(ci95(all_on), 4),
        "skill_acc_all_on": round(all_on, 4),
        "skill_acc_experts_off": round(experts_off, 4),
        "skill_acc_shared_ffn_off": round(ffn_off, 4),
        "skill_acc_embedding_attention_only": round(resident_only, 4),
        "gap_experts": round(gap_experts, 4),
        "gap_shared_ffn": round(gap_shared_ffn, 4),
        "fact_acc_kb_on": round(kb_on["acc"], 4),
        "fact_acc_kb_off": round(kb_off["acc"], 4),
        "gap_kb": round(gap_kb, 4),
        "fact_retrieval_hit": round(kb_on.get("retrieval_hit", 0.0), 4),
        "fact_acc_upper_bound_handwritten": round(upper["acc"], 4),
        "presentation_gap": round(upper["acc"] - kb_on["acc"], 4),
        "facts_also_in_weights": dup,
        "facts_conflicting_with_kb": conflict,
        "prior_wrong_on_held_facts": prior_wrong,
        "evidence_reliance": round(reliance, 4),
        "evidence_makes_it_worse": hurt,
        "kb_size": kb.size(),
    }

    verdicts = []
    if gap_experts <= 0.0:
        verdicts.append("EXPERTS_EMPTY：专家清零不影响任何答案——本次训练没有产生专家知识，"
                        "任何关于动态 MoE（驻留/数量）的结论都不成立")
    if resident_only >= 0.30:
        verdicts.append(f"KNOWLEDGE_IN_RESIDENT_PART：把常识 FFN 和全部专家都关掉后仍有 "
                        f"{resident_only*100:.0f}%（只剩 embedding + 注意力）——模型会答的东西住在"
                        "必须常驻的那部分里。这种任务上 MoE 没有任何可调度的知识，"
                        "换任何调度策略都不会改变答案")
    if gap_kb > 0.0:
        verdicts.append(f"KB_CARRIES_FACTS：事实题 检索开 {kb_on['acc']*100:.0f}% / "
                        f"检索关 {kb_off['acc']*100:.0f}%，这些知识确实在知识库")
    if upper["acc"] - kb_on["acc"] > 0.03:
        verdicts.append(f"PRESENTATION_LIMIT：检索命中 {kb_on.get('retrieval_hit', 0)*100:.0f}% "
                        f"但只有 {kb_on['acc']*100:.0f}%，手抄进题面能到 "
                        f"{upper['acc']*100:.0f}%——差距在检索片段的呈现格式，不在检索")
    if conflict:
        verdicts.append(f"PRIOR_CONFLICTS_WITH_KB：{conflict} 条事实模型自己给了一个错的服务名，"
                        "KB 里是另一个——同一条知识有两个权威来源，更新 KB 未必生效")
    if reliance < 0.9:
        verdicts.append(f"PRIOR_OVER_EVIDENCE：先验答错 {prior_wrong} 条，其中只有 {followed} 条"
                        f"采纳了 KB 的值（证据优先率 {reliance*100:.0f}%）——已验证的知识条目"
                        "压不过模型自己的先验。这就是准确率上界不是 100% 的原因，"
                        "和检索、和呈现格式都无关")
    if hurt:
        verdicts.append(f"EVIDENCE_HURTS：{hurt} 条本来先验就是对的，注入检索反而答错——"
                        "注入的干扰项有代价，考虑只注入 rank-1")
    if dup:
        verdicts.append(f"IN_WEIGHTS_TOO：{dup} 条事实不用检索也答对，KB 与权重同时持有")
    out["verdicts"] = verdicts

    print(f"\n== 知识归属探针 ==  检查点 {ckpt}  留出技能题 {len(skill)} 题"
          f"（95% 置信半宽 ±{ci95(all_on)*100:.1f}pp，只有 0 题变化是确定的）")
    print(f"  全部开 {all_on*100:.1f}%   专家清零 {experts_off*100:.1f}% "
          f"(gap {gap_experts*100:+.1f}pp)   常识 FFN 清零 {ffn_off*100:.1f}% "
          f"(gap {gap_shared_ffn*100:+.1f}pp)")
    print(f"  两个 FFN 都关掉，只剩 embedding + 注意力：{resident_only*100:.1f}%"
          f"——这部分必须常驻，MoE 动不了它")
    print(f"  知识库：检索开 {kb_on['acc']*100:.0f}%（命中 {kb_on.get('retrieval_hit',0)*100:.0f}%）"
          f"  检索关 {kb_off['acc']*100:.0f}%   gap_kb = {gap_kb*100:+.0f}pp")
    print(f"  上界（答案手抄进题面）：{upper['acc']*100:.0f}%   "
          f"presentation_gap = {(upper['acc']-kb_on['acc'])*100:+.0f}pp")
    print(f"  不用检索也答对的事实 {dup} 条，其中与 KB 冲突 {conflict} 条")
    print(f"  证据优先率：先验答错 {prior_wrong} 条 → 采纳 KB 的 {followed} 条 = "
          f"{reliance*100:.0f}%；先验本来对、被检索带错的 {hurt} 条")
    print("\n判定：")
    for v in verdicts or ["（无）"]:
        print(f"  · {v}")

    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
        print(f"\n写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
