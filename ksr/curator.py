"""让常驻常识专家当知识库的策展人。

你提的那条：出错被修正之后，"要不要写进知识库、写成什么、跟已有条目冲突怎么办"
由常驻的常识专家判断和执行，而不是每次都靠人审。它常驻显存，所以这件事
**不增加任何加载成本**——这是这个设计最划算的地方。

但它有一个真实风险：同一个模块既负责答题、又负责裁定什么为真，
容易把自己的错当成新知识写进去（自我确认偏误）。所以这里用两条硬约束兜住，
**写死在代码里，不依赖模型输出**：

  1. 没有出处的纠正不允许凭空新增（只能改/合并/否决）；
  2. 与已验证条目语义冲突时，必须显式标 conflict 并交人工，
     不允许静默覆盖——静默覆盖会让知识库被错误纠正越改越坏。

三种动作：0=拒绝写入，1=新增，2=合并到已有条目（命中计数 +1，不新增行）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSRConfig
from .knowledge import KnowledgeBase, KnowledgeRecord
from .nets import KSRModel
from .tokenizer import CharTokenizer

ACTION_REJECT, ACTION_ADD, ACTION_MERGE = 0, 1, 2
ACTION_NAME = {ACTION_REJECT: "拒绝", ACTION_ADD: "新增", ACTION_MERGE: "合并"}


@torch.no_grad()
def text_repr(model: KSRModel, tok: CharTokenizer, text: str) -> torch.Tensor:
    """一段文本的表示：词元嵌入的均值（再取 RMS 归一）。不额外跑前向，几乎零成本。"""
    ids = tok.encode(text)
    if not ids:
        return torch.zeros(model.cfg.hidden_size)
    h = model.embed(torch.tensor([ids])) * model.embed_scale
    return model.norm(h)[0].mean(0).contiguous()


class KnowledgeCurator(nn.Module):
    """输入四路表示 → 动作 + 冲突标志。参数量 ~0.1M，挂在常识专家的输出侧。"""

    def __init__(self, cfg: KSRConfig, inner: int = 256):
        super().__init__()
        h = cfg.hidden_size
        self.in_proj = nn.Linear(4 * h, inner, bias=False)
        self.act = nn.SiLU()
        self.head_action = nn.Linear(inner, 3, bias=False)
        self.head_conflict = nn.Linear(inner, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.act(self.in_proj(feats))
        return self.head_action(z), self.head_conflict(z).squeeze(-1)

    # 四路特征：查询 / 错误答案 / 纠正答案 / 检索到的相关条目（没有就填零）
    def feats(self, model, tok, query: str, wrong: str, corrected: str,
              retrieved: str = "") -> torch.Tensor:
        q = text_repr(model, tok, query)
        w = text_repr(model, tok, wrong)
        c = text_repr(model, tok, corrected)
        r = text_repr(model, tok, retrieved) if retrieved else torch.zeros_like(q)
        return torch.cat([q, w, c, r])


@dataclass
class Curation:
    action: int
    action_name: str
    conflict: bool
    merge_into: Optional[str]      # 合并目标条目的 rid
    reason: str
    confidence: float


def curate(model: KSRModel, tok: CharTokenizer, curator: KnowledgeCurator,
           kb: KnowledgeBase, query: str, wrong: str, corrected: str,
           *, source: str, domains: Sequence[str], top_k: int = 3,
           sim_merge: float = 0.82, sim_conflict: float = 0.55) -> Curation:
    """先由检索给出"已有条目"，再由常识专家决定动作；硬约束在这里兜底。"""
    qv = text_repr(model, tok, query)
    hits: List[Tuple[float, KnowledgeRecord]] = []
    for d in domains:
        hits += kb.indices[d].search(qv, top_k) if d in kb.indices else []
    hits.sort(key=lambda x: -x[0])
    best_score, best = hits[0] if hits else (0.0, None)

    feats = curator.feats(model, tok, query, wrong, corrected,
                         best.text if best is not None else "")
    with torch.no_grad():
        a_logits, c_logit = curator(feats.unsqueeze(0))
    action = int(a_logits[0].argmax())
    conflict_model = float(torch.sigmoid(c_logit[0]))
    confidence = float(torch.softmax(a_logits[0], dim=-1)[action])

    # ── 硬约束 1：与已验证条目同义 → 只能合并，不能新增（否则知识库越长越臃肿）
    if best is not None and best_score >= sim_merge and best.answer.strip() == corrected.strip():
        return Curation(ACTION_MERGE, ACTION_NAME[ACTION_MERGE], False, best.rid,
                       f"检索到同义已验证条目（相似度 {best_score:.2f}）→ 合并，命中计数 +1",
                       1.0)
    # ── 硬约束 2：与已验证条目矛盾 → 显式标冲突，交人工，不静默覆盖
    if (best is not None and best.verified and best_score >= sim_conflict
            and best.answer.strip() and best.answer.strip() != corrected.strip()):
        return Curation(ACTION_REJECT, ACTION_NAME[ACTION_REJECT], True, best.rid,
                       f"与已验证条目冲突（相似度 {best_score:.2f}，已有答案 {best.answer!r}，"
                       f"纠正答案 {corrected!r}）→ 标记冲突交人工，不覆盖",
                       max(conflict_model, best_score))
    # ── 硬约束 3：没有出处的纠正不允许凭空新增
    if action == ACTION_ADD and source in ("", "model", "self"):
        return Curation(ACTION_REJECT, ACTION_NAME[ACTION_REJECT], False, None,
                       "纠正没有出处（模型自己生成的纠正不能自证为真）→ 拒绝",
                       confidence)
    if action == ACTION_ADD:
        return Curation(ACTION_ADD, ACTION_NAME[ACTION_ADD], False, None,
                       f"无相关已验证条目（最高相似度 {best_score:.2f}），出处={source} → 新增",
                       confidence)
    if best is not None:
        return Curation(ACTION_MERGE, ACTION_NAME[ACTION_MERGE], False, best.rid,
                       "模型判断为补充信息 → 合并到最相近条目", confidence)
    return Curation(ACTION_REJECT, ACTION_NAME[ACTION_REJECT], False, None,
                    "既没有可合并对象也没有出处 → 拒绝", confidence)


def build_training_pairs(model: KSRModel, tok: CharTokenizer, kb: KnowledgeBase,
                         facts: dict, rng) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造四种情形的训练对（这就是"它凭什么知道该拒绝"的来源，不是拍脑袋阈值）：

      A 检索到同一条且答案一致      → 合并
      B 检索到同一条但答案矛盾      → 拒绝 + 标冲突
      C 什么都没检索到              → 新增
      D 检索到的是无关问题          → 新增（相似度低，不算冲突）
    """
    items = list(facts.items())
    X, Y, C = [], [], []

    def push(query, wrong, corr, retr, action, conflict):
        X.append(curator_feats_only(model, tok, query, wrong, corr, retr))
        Y.append(action)
        C.append(float(conflict))

    for _ in range(len(items) * 2):
        port, svc = items[rng.randrange(len(items))]
        q = f"查询端口 {port} 上运行的服务是什么"
        other = svc
        while other == svc:
            other = items[rng.randrange(len(items))][1]
        entry = f"端口 {port} 的服务是 {svc}"
        # A：已有正确条目 → 合并
        push(q, other, svc, entry, ACTION_MERGE, 0)
        # B：已有条目答案相反 → 拒绝 + 冲突
        push(q, svc, other, entry, ACTION_REJECT, 1)
        # C：没有检索结果 → 新增
        push(q, other, svc, "", ACTION_ADD, 0)
        # D：检索到的是别的问题 → 新增
        p2 = items[rng.randrange(len(items))][0]
        push(q, other, svc, f"端口 {p2} 的服务是 {items[rng.randrange(len(items))][1]}",
             ACTION_ADD, 0)

    return (torch.stack(X), torch.tensor(Y), torch.tensor(C, dtype=torch.float32))


def curator_feats_only(model, tok, q, w, c, r):
    h = model.cfg.hidden_size
    parts = [text_repr(model, tok, q), text_repr(model, tok, w), text_repr(model, tok, c)]
    parts.append(text_repr(model, tok, r) if r else torch.zeros(h))
    return torch.cat(parts)


def train_curator(model: KSRModel, tok: CharTokenizer, curator: KnowledgeCurator,
                  kb: KnowledgeBase, facts: dict, steps: int = 120, bs: int = 32,
                  lr: float = 3e-3, seed: int = 0) -> dict:
    """只训练策展人自己的头，不动模型主干——知识判断不该把语言模型一起改坏。"""
    rng = __import__("random").Random(seed)
    X, Y, C = build_training_pairs(model, tok, kb, facts, rng)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(curator.parameters(), lr=lr)
    n = len(X)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        a_logits, c_logit = curator(X[idx])
        loss = (F.cross_entropy(a_logits, Y[idx])
                + F.binary_cross_entropy_with_logits(c_logit, C[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        if step % 30 == 0 or step == steps - 1:
            acc = (a_logits.argmax(-1) == Y[idx]).float().mean().item()
            print(f"  curator step {step:3d}  loss {float(loss.detach()):.3f}  acc {acc:.2f}")
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return {"loss_first": hist[0], "loss_last": hist[-1],
            "trainable_params": sum(p.numel() for p in curator.parameters())}
