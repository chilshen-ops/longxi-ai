"""自我进化的两条路径。

  即时回退（不改权重）：路由置信度决定走哪条路——
      L0 高置信 → 主专家，按需检索
      L1 中置信 → 主专家 + 常识校验器复核，判错就换候选专家重试
      L2 低置信 → 只走常识通路（强制 general 专家）+ 必须检索
    这就是"选错专家不会错到底"的机制，而不是事后人工兜底。

  参数沉淀（改权重）：审核通过的纠正案例 → 对应专家的小步微调 →
    下次同类问题**不检索也能答对**。这才叫"自我迭代"，
    同时必须测"该专家其他题有没有被带坏"（防遗忘），否则沉淀就是自毁。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import KSRConfig
from .knowledge import CorrectionRecord
from .nets import KSRModel
from .tokenizer import CharTokenizer


# ── 即时回退：三级路由策略 ─────────────────────────────────────────────────────
@dataclass
class Decision:
    expert: int
    need_knowledge: bool
    level: str                  # L0 | L1 | L2
    confidence: float
    reason: str


def route_policy(agg: Dict, cfg: KSRConfig) -> Decision:
    """把请求级路由信号变成一个可审计的决定（带理由，便于线上排查）。"""
    conf = float(agg["confidence"])
    need = float(agg["need_knowledge"]) >= 0.5
    primary = int(agg["primary_expert"])
    if conf < cfg.router_confidence_low:
        return Decision(expert=cfg.general_expert, need_knowledge=True, level="L2",
                        confidence=conf,
                        reason=f"置信 {conf:.2f} < {cfg.router_confidence_low}：只走常识通路并强制检索")
    if conf < cfg.router_confidence_high:
        return Decision(expert=primary, need_knowledge=True, level="L1", confidence=conf,
                        reason=f"置信 {conf:.2f} 处于中间带：主专家 + 常识校验")
    return Decision(expert=primary, need_knowledge=need, level="L0", confidence=conf,
                    reason=f"置信 {conf:.2f} 高：直接走主专家")


def predict(model: KSRModel, tok: CharTokenizer, prompt: str,
            force_expert: Optional[int] = None, max_new: int = 24) -> Tuple[str, Dict]:
    ids = tok.encode_prompt(prompt)       # 与训练侧同一条编码路径
    out, agg = model.generate(ids, max_new=max_new, force_expert=force_expert)
    return tok.decode(out).strip(), agg


def _verdict(model: KSRModel, tok: CharTokenizer, prompt: str,
             force: Optional[int]) -> Tuple[bool, float]:
    """在"A: 之后那个位置"让常识校验器表态：(是否判错, 判对概率)。"""
    ids = tok.encode_prompt(prompt)
    X = torch.tensor([ids])
    with torch.no_grad():
        _, meta = model(X, force_expert=force)
        h = model.norm(meta["final_hidden"])
        r = meta["reasoner_ref"]
        _, probs = model.corrector.verify(h[:, len(ids) - 1], r[:, len(ids) - 1])
    p = probs[0]
    return bool(int(p.argmax()) == 2), float(p[0])


def answer_with_correction(model: KSRModel, tok: CharTokenizer, prompt: str,
                           cfg: KSRConfig, candidates: Sequence[int],
                           max_tries: int = 3) -> Dict:
    """L0/L1/L2 主流程：先按策略走，校验器判错就在候选专家里重试。"""
    _, agg = predict(model, tok, prompt)
    d = route_policy(agg, cfg)
    forwards = 1
    if d.level == "L2":
        pred, agg = predict(model, tok, prompt, force_expert=d.expert)
        return {"answer": pred, "decision": d, "forwards": forwards, "recovered": False}

    pred, _ = predict(model, tok, prompt, force_expert=d.expert)
    if d.level == "L0":
        return {"answer": pred, "decision": d, "forwards": forwards, "recovered": False}

    wrong, _ = _verdict(model, tok, prompt, d.expert)
    if not wrong:
        return {"answer": pred, "decision": d, "forwards": forwards, "recovered": False}
    best, best_score = pred, -1.0
    for cand in candidates:
        cand = int(cand)
        if cand == d.expert:
            continue
        p, _ = predict(model, tok, prompt, force_expert=cand)
        _, ok = _verdict(model, tok, prompt, cand)
        forwards += 1
        if ok > best_score:
            best, best_score = p, ok
        if forwards >= max_tries:
            break
    return {"answer": best, "decision": d, "forwards": forwards, "recovered": True}


# ── 参数沉淀：L2 微调 ─────────────────────────────────────────────────────────
def corrections_to_examples(records: Sequence[CorrectionRecord]) -> List[Tuple[str, str, str]]:
    """审核通过的纠正 → (prompt, answer, domain) 微调样本。

    注意存的是"问题 + 正确答案"，不是把错误答案也塞进去——
    错误答案只进知识库当负例，不进微调集，否则会把错误再学一遍。
    """
    out = []
    for r in records:
        q = r.query if r.query.endswith("\nA: ") else r.query.rstrip() + "\nA: "
        out.append((q, r.corrected_output.strip(), r.domain_tag or "general"))
    return out


@torch.no_grad()
def accuracy(model: KSRModel, tok: CharTokenizer, items: Sequence[Tuple[str, str, str]]) -> float:
    model.eval()
    ok = 0
    for prompt, gold, *_ in items:
        pred, _ = predict(model, tok, prompt)
        ok += int(pred == gold)
    return ok / max(1, len(items))


def finetune_expert(model: KSRModel, tok: CharTokenizer, cfg: KSRConfig, expert_id: int,
                    examples: Sequence[Tuple[str, str, str]], steps: int = 60,
                    lr: float = 8e-4) -> Dict:
    """只解冻这一个专家的适配器栈，其余全部冻结。

    这正是"专家解耦"带来的能力：改一个技能不动其他技能，也不动常识底座。
    """
    stack = model.experts.pool[expert_id]
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [p for p in stack.parameters()]
    for p in trainable:
        p.requires_grad_(True)

    def encode(prompt: str, answer: str):
        p = tok.encode_prompt(prompt)     # 含 BOS，与推理侧一致
        a = tok.encode(answer) + [tok.eos_id]
        ids = p + a
        labels = [-100] * len(p) + a
        assert len(ids) == len(labels)
        return ids, labels

    n = max(len(encode(p, a)[0]) for p, a, *_ in examples)
    X = torch.full((len(examples), n), tok.pad_id, dtype=torch.long)
    Y = torch.full((len(examples), n), -100, dtype=torch.long)
    for i, (p, a, *_) in enumerate(examples):
        xs, ys = encode(p, a)
        X[i, :len(xs)] = torch.tensor(xs)
        Y[i, :len(ys)] = torch.tensor(ys)

    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        logits, _ = model(X, force_expert=expert_id)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                               Y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "trainable_params": sum(p.numel() for p in trainable)}
