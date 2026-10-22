"""KSR-MoE 网络本体：混合注意力 + 全局路由 + 常识专家常驻 + 低秩技能专家旁路。

与"贴出的原稿"相比，这里修掉了四个会让结论失效的实现问题：
  1. 全注意力带因果掩码（原稿 prefill 无掩码 → 自回归语义错误、与 decode 不一致）
  2. 路由置信度取**全量 softmax 的 top-1**（原稿 top-1 后再 softmax 恒等于 1，
     门控拿不到梯度、阈值永远失效）
  3. 专家是"24 层一组"的适配器栈，按整栈加载/卸载（原稿把 layer0 的专家塞进
     全局 cache，导致所有层都用 layer0 的权重）
  4. 纠错器只有一份、只在请求级触发（原稿每层一个、每 token 必跑，1.2B 参数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KSRConfig


# ── 基础件 ────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float, device=None) -> torch.Tensor:
    """返回 [max_len, head_dim//2] 的角度，供半旋转式 RoPE 使用。"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    return torch.outer(t, inv)


def apply_rotary(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, Dh]; angles: [S, Dh//2]"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── 注意力 ────────────────────────────────────────────────────────────────────
@dataclass
class LayerCache:
    """全注意力层持有 KV；线性注意力层持有恒定大小的状态（与序列长度无关）。"""
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None
    length: int = 0


class CacheList:
    def __init__(self, num_layers: int):
        self.slots: List[LayerCache] = [LayerCache() for _ in range(num_layers)]

    def __getitem__(self, i: int) -> LayerCache:
        return self.slots[i]


class GQASelfAttention(nn.Module):
    """GQA（原稿误称 MQA）+ RoPE + 因果掩码 + KV Cache。"""

    def __init__(self, cfg: KSRConfig):
        super().__init__()
        self.h = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_kv_groups
        self.dh = cfg.head_dim
        self.groups = self.n_heads // self.n_kv
        self.q_proj = nn.Linear(self.h, self.n_heads * self.dh, bias=False)
        self.k_proj = self.v_proj = None
        self.k_proj = nn.Linear(self.h, self.n_kv * self.dh, bias=False)
        self.v_proj = nn.Linear(self.h, self.n_kv * self.dh, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.dh, self.h, bias=False)

    def _shape(self, w: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, s, _ = w.shape
        return w.view(b, s, n_heads, self.dh).transpose(1, 2)

    def forward(self, x: torch.Tensor, angles: torch.Tensor,
                cache: Optional[LayerCache] = None, start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        q = apply_rotary(self._shape(self.q_proj(x), self.n_heads), angles[start_pos:start_pos + s])
        k = apply_rotary(self._shape(self.k_proj(x), self.n_kv), angles[start_pos:start_pos + s])
        v = self._shape(self.v_proj(x), self.n_kv)

        if cache is not None:
            if cache.k is not None:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v, cache.length = k, v, k.shape[2]

        if self.groups > 1:  # GQA：KV 头复制到每个 query 头
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if cache is not None and s == 1:
            # 增量解码：单 query 看全部历史，不需要掩码
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.dh)
        return self.o_proj(out)


class GatedDeltaNet(nn.Module):
    """线性注意力（Gated DeltaNet 参考实现，O(n)）。

    为可读性按时间步递推；生产实现应换成 chunked/welford kernel。
    状态大小 = heads × dk × dv，与序列长度无关 —— 这是混合注意力的主要显存收益。
    """

    def __init__(self, cfg: KSRConfig):
        super().__init__()
        la = cfg.linear_attn
        self.n_heads = la.num_v_heads
        self.dk = la.head_dim_k
        self.dv = la.head_dim_v
        self.n_kv = la.num_kv_groups
        self.h = cfg.hidden_size
        self.q_proj = nn.Linear(self.h, self.n_heads * self.dk, bias=False)
        # 与全注意力层同预算：共享 K/V 投影（原型口径，见 SPEC §2 假设）
        self.k_proj = nn.Linear(self.h, self.n_kv * self.dk, bias=False)
        self.v_proj = nn.Linear(self.h, self.n_kv * self.dv, bias=False)
        self.g_proj = nn.Linear(self.h, self.n_heads, bias=True)      # 衰减门
        self.b_proj = nn.Linear(self.h, self.n_heads, bias=True)      # 写入门
        if self.n_kv != self.n_heads:
            # 头数不同时需要把共享的 K/V 映射到每个头
            self.k_expand = nn.Linear(self.n_kv * self.dk, self.n_heads * self.dk, bias=False)
            self.v_expand = nn.Linear(self.n_kv * self.dv, self.n_heads * self.dv, bias=False)
        else:
            self.k_expand = self.v_expand = None
        self.o_proj = nn.Linear(self.n_heads * self.dv, self.h, bias=False)

    def forward(self, x: torch.Tensor, cache: Optional[LayerCache] = None,
                start_pos: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        dev, dt = x.device, x.dtype
        q = self.q_proj(x).view(b, s, self.n_heads, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.dv).transpose(1, 2)
        if self.k_expand is not None:
            k = self.k_expand(k.transpose(1, 2).reshape(b, s, -1)).view(b, s, self.n_heads, self.dk).transpose(1, 2)
            v = self.v_expand(v.transpose(1, 2).reshape(b, s, -1)).view(b, s, self.n_heads, self.dv).transpose(1, 2)
        g = torch.sigmoid(self.g_proj(x)).transpose(1, 2)   # [B,H,S]
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        state = cache.state if (cache is not None and cache.state is not None) else \
            q.new_zeros(b, self.n_heads, self.dk, self.dv)
        outs = []
        for t in range(s):
            kt, vt = k[:, :, t], v[:, :, t]                 # [B,H,Dk],[B,H,Dv]
            state = state * g[:, :, t][..., None, None]     # 衰减
            pred = torch.einsum("bhk,bhkv->bhv", kt, state)  # 读状态
            delta = vt - pred                                 # delta rule
            state = state + torch.einsum("bhk,bhv->bhkv", kt, delta * beta[:, :, t][..., None])
            o = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
            outs.append(o)
        out = torch.stack(outs, dim=2).transpose(1, 2).reshape(b, s, self.n_heads * self.dv)
        if cache is not None:
            cache.state, cache.length = state, cache.length + s
        return self.o_proj(out.to(dt))


# ── 常识专家（常驻）与技能专家（低秩旁路）──────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertStore(nn.Module):
    """技能专家池：每个专家 = 每层一个低秩旁路（h→r→h），**按整栈加载/卸载**。

    参数以张量形式集中存放（不是 nn.ModuleList 里的一堆模块），这样
    "卸载"就是把张量搬到 CPU/磁盘，不会把 layer0 的权重误当成全局权重。
    """

    def __init__(self, cfg: KSRConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.expert_rank
        self.alpha = float(r)  # 旁路缩放：out += (alpha/r) * down(silu(gate(x))*up(x))
        pool = []
        for _ in range(cfg.num_skill_experts):
            per_layer = nn.ModuleList([SwiGLU(cfg.hidden_size, r) for _ in range(cfg.num_layers)])
            pool.append(per_layer)
        self.pool = nn.ModuleList(pool)
        self.hot: List[int] = list(range(min(cfg.hot_experts, cfg.num_skill_experts)))
        self.resident_bytes = 0
        self._update_accounting()
        self.load_events = 0
        self.evict_events = 0

    def _update_accounting(self) -> None:
        per = sum(p.numel() for p in self.pool[0].parameters()) if self.cfg.num_skill_experts else 0
        self.stack_bytes = per * self.cfg.num_layers
        self.resident_bytes = self.stack_bytes * len(self.hot) * self._itemsize()

    def _itemsize(self) -> int:
        for p in self.parameters():
            return p.element_size()
        return 4

    def set_hot(self, hot: List[int]) -> None:
        """热集合：只有这些专家的栈留在计算设备上；其余按需搬运（LRU）。"""
        hot = [e for e in hot if 0 <= e < self.cfg.num_skill_experts]
        for eid in range(self.cfg.num_skill_experts):
            device = self.compute_device if eid in hot else torch.device("cpu")
            with torch.no_grad():
                for p in self.pool[eid].parameters():
                    p.data = p.data.to(device)
        self.hot = list(hot)
        self._update_accounting()

    compute_device: torch.device = torch.device("cpu")

    def ensure_hot(self, eid: int) -> bool:
        """把某个专家的整栈搬上计算设备。返回是否发生了搬运（用于统计命中率）。"""
        if eid in self.hot:
            return False
        if len(self.hot) >= self.cfg.hot_experts:
            victim = self.hot.pop(0)
            with torch.no_grad():
                for p in self.pool[victim].parameters():
                    p.data = p.data.to("cpu")
            self.evict_events += 1
        with torch.no_grad():
            for p in self.pool[eid].parameters():
                p.data = p.data.to(self.compute_device)
        self.hot.append(eid)
        self.load_events += 1
        self._update_accounting()
        return True


# ── 路由 ──────────────────────────────────────────────────────────────────────
class Router(nn.Module):
    """输出：专家 id / 混合权重 / 置信度 / 是否需要知识 / 负载均衡辅助损失。

    置信度 = 全量 softmax 的 top-1 概率（可校准、可用于阈值），
    而不是"top-k 之后再 softmax"——那样 top-1 时恒等于 1。
    """

    def __init__(self, cfg: KSRConfig):
        super().__init__()
        self.top_k = cfg.top_k_experts
        self.n_exp = cfg.num_skill_experts
        self.gate = nn.Linear(cfg.hidden_size, cfg.router_hidden, bias=False)
        self.expert_head = nn.Linear(cfg.router_hidden, self.n_exp, bias=False)
        self.knowledge_head = nn.Linear(cfg.router_hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.expert_head(F.silu(self.gate(x)))          # [B,S,E]
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        weights = top_p / top_p.sum(-1, keepdim=True)            # top-k 归一（Switch 口径）
        confidence = top_p[:, :, 0]                              # 全量 softmax 的 top-1
        need = torch.sigmoid(self.knowledge_head(F.silu(self.gate(x)))).squeeze(-1)

        # Switch 风格负载均衡：f_e 不可导但 P_e 可导，梯度经 P_e 回传
        with torch.no_grad():
            f = F.one_hot(top_i[:, :, 0], self.n_exp).float().mean(dim=(0, 1))
        p_mean = probs.mean(dim=(0, 1))
        aux = self.n_exp * torch.sum(f * p_mean)
        return {"expert_ids": top_i, "expert_weights": weights, "confidence": confidence,
                "need_knowledge": need, "aux_loss": aux}


# ── 序列级纠错器（单份，请求级触发）────────────────────────────────────────────
class SequenceCorrector(nn.Module):
    def __init__(self, cfg: KSRConfig):
        super().__init__()
        h = cfg.hidden_size
        c = cfg.corrector
        self.verifier = nn.Sequential(nn.Linear(2 * h, c.hidden, bias=False), nn.SiLU(),
                                      nn.Linear(c.hidden, c.num_classes, bias=False))
        self.correction_proj = nn.Linear(h, h, bias=False) if c.with_correction_proj else None

    def verify(self, hidden: torch.Tensor, reasoner_out: torch.Tensor):
        logits = self.verifier(torch.cat([hidden, reasoner_out], dim=-1))
        return logits.argmax(-1), F.softmax(logits, dim=-1)

    def correct(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.correction_proj is not None
        return self.correction_proj(hidden)


# ── KSR 层 ────────────────────────────────────────────────────────────────────
class KSRBlock(nn.Module):
    def __init__(self, cfg: KSRConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.is_full = cfg.is_full_attention(layer_index)
        self.attn = GQASelfAttention(cfg) if self.is_full else GatedDeltaNet(cfg)
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.router = Router(cfg)
        self.reasoner = SwiGLU(cfg.hidden_size, cfg.reasoner_hidden)   # 常驻，永不卸载
        self.reasoner_norm = RMSNorm(cfg.hidden_size)
        self.expert_norm = RMSNorm(cfg.hidden_size)
        self.top_k = cfg.top_k_experts

    def forward(self, x: torch.Tensor, experts: ExpertStore, angles: torch.Tensor,
                cache: Optional[LayerCache] = None, start_pos: int = 0,
                force_expert: Optional[int] = None) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), angles, cache, start_pos) if self.is_full \
            else x + self.attn(self.attn_norm(x), cache, start_pos)

        route = self.router(x)
        reasoner_out = self.reasoner(self.reasoner_norm(x))
        x = x + reasoner_out                                            # 常识专家：始终激活

        ids, w = route["expert_ids"], route["expert_weights"]
        b, s, k = ids.shape
        expert_out = torch.zeros_like(x)
        if force_expert is not None:
            # 强制走某一个专家：不看路由选的是谁。
            # 原来这里只在"路由恰好选中 eid"的位置上套 target，于是 target 与路由不一致时
            # 一个专家都没用上——测出来的"选错专家"其实是"完全不用专家"，
            # 而 L2 微调则因为专家压根没进计算图而报 "does not require grad"。
            # 权重仍沿用路由给的分，保证强制路径和普通路径的量级可比。
            sel = self.expert_norm(x)
            out = experts.pool[force_expert][self.layer_index - 1](sel)
            expert_out = out * w.sum(-1, keepdim=True)
        else:
            for slot in range(k):
                for eid in range(experts.cfg.num_skill_experts):
                    mask = (ids[:, :, slot] == eid)
                    if not mask.any():
                        continue
                    sel = self.expert_norm(x)[mask]                        # [N,H] 一次性批量算
                    out = experts.pool[eid][self.layer_index - 1](sel)     # 整栈中的第 layer 片
                    expert_out[mask] += out * w[mask, slot].unsqueeze(-1)
        x = x + expert_out
        return x, route, reasoner_out


# ── 完整模型 ──────────────────────────────────────────────────────────────────
class KSRModel(nn.Module):
    def __init__(self, cfg: KSRConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        # tied embedding 若用 PyTorch 默认初始化（std=1），logits 数量级会到 sqrt(h) 以上，
        # 初始 loss 直接飙到 200+（正常应约 ln(V)≈6），训练又慢又不稳。
        # 按 GPT-2 的做法：权重 std=0.02，输入侧再乘 sqrt(h) 把信号补回来，输出 logits 保持小数量级。
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.embed_scale = float(cfg.hidden_size) ** 0.5
        self.blocks = nn.ModuleList([KSRBlock(cfg, i + 1) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.embed.weight
        self.experts = ExpertStore(cfg)
        self.corrector = SequenceCorrector(cfg) if cfg.corrector.enabled else None
        self.register_buffer("rope", precompute_rope(cfg.head_dim, cfg.context_len, cfg.rope_theta),
                             persistent=False)

    # 输出投影的计算量与权重是否绑定无关（绑定省字节，不省矩阵乘）
    def activated_params(self) -> int:
        c = self.cfg
        per_layer = sum(sum(p.numel() for p in blk.parameters()) for blk in self.blocks)
        active_expert = sum(p.numel() for p in self.experts.pool[0].parameters()) * c.top_k_experts
        return per_layer + active_expert + c.vocab_size * c.hidden_size + c.hidden_size

    def forward(self, input_ids: torch.Tensor, cache: Optional[CacheList] = None,
                start_pos: int = 0, force_expert: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        h = self.embed(input_ids) * self.embed_scale
        meta: Dict = {"confidence": [], "need_knowledge": [], "expert_ids": [], "aux_loss": 0.0}
        last_reasoner = None
        for i, blk in enumerate(self.blocks):
            slot = cache[i] if cache is not None else None
            h, route, last_reasoner = blk(h, self.experts, self.rope, slot, start_pos, force_expert)
            meta["confidence"].append(route["confidence"])
            meta["need_knowledge"].append(route["need_knowledge"])
            meta["expert_ids"].append(route["expert_ids"])
            meta["aux_loss"] = meta["aux_loss"] + route["aux_loss"]
        meta["aux_loss"] = meta["aux_loss"] / max(1, self.cfg.num_layers)
        meta["final_hidden"] = h
        meta["reasoner_ref"] = last_reasoner
        return self.lm_head(self.norm(h)), meta

    # ── 请求级信号聚合：用 min/p5，不用 mean ────────────────────────────────
    @staticmethod
    def aggregate(meta: Dict) -> Dict:
        conf = torch.stack(meta["confidence"]).amin(dim=(0, 2))      # 跨层跨 token 取最小
        need = torch.stack(meta["need_knowledge"]).amax(dim=(0, 2))  # 任一层认为需要就检索
        ids = torch.stack(meta["expert_ids"])[:, :, :, 0]
        mode = torch.mode(ids.reshape(-1)).values.item()             # 请求级主专家（众数）
        return {"confidence": float(conf.min()), "need_knowledge": float(need.max()),
                "primary_expert": int(mode)}

    @torch.no_grad()
    def generate(self, input_ids: List[int], max_new: int = 32, temperature: float = 0.0,
                 force_expert: Optional[int] = None) -> Tuple[List[int], Dict]:
        self.eval()
        ids = list(input_ids)
        cur = torch.tensor([ids], device=self.embed.weight.device)
        cache = CacheList(self.cfg.num_layers)
        logits, meta = self.forward(cur, cache, 0, force_expert)
        nxt = logits[:, -1, :]
        out = []
        for step in range(max_new):
            if temperature > 0:
                probs = F.softmax(nxt / temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
            else:
                token = int(nxt.argmax(-1).item())
            if token == self.cfg_eos_id:
                break
            out.append(token)
            cur = torch.tensor([[token]], device=nxt.device)
            logits, _ = self.forward(cur, cache, len(ids) + step, force_expert)
            nxt = logits[:, -1, :]
        return out, self.aggregate(meta)

    cfg_eos_id: int = 2
