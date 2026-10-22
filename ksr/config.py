"""配置：JSON 是单一真源，模型和计算器读同一份。

任何一处规格改动都必须落到 JSON 里，否则 tests/test_spec_consistency.py 会失败。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CorrectorConfig:
    enabled: bool = False
    placement: str = "sequence"
    hidden: int = 0
    num_classes: int = 3
    with_correction_proj: bool = True
    always_on: bool = False
    # 触发策略（请求级）：低于该分位/阈值才进入纠错，避免"每 token 都跑"
    token_wrong_ratio: float = 0.25
    confidence_low: float = 0.5


@dataclass
class LinearAttnConfig:
    num_kv_groups: int = 4
    num_v_heads: int = 32
    head_dim_k: int = 128
    head_dim_v: int = 128


@dataclass
class KSRConfig:
    name: str = "ksr-tiny"
    vocab_size: int = 0
    hidden_size: int = 256
    num_layers: int = 2
    num_attention_heads: int = 4
    num_kv_groups: int = 2
    head_dim: int = 64
    full_attn_layers: List[int] = field(default_factory=lambda: [1])
    linear_attn: LinearAttnConfig = field(default_factory=LinearAttnConfig)
    ffn_type: str = "swiglu"
    reasoner_hidden: int = 512
    num_skill_experts: int = 4
    expert_rank: int = 64
    expert_ffn_type: str = "swiglu"
    top_k_experts: int = 1
    router_hidden: int = 128
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    tied_embeddings: bool = True
    context_len: int = 512
    rope_theta: float = 10_000.0
    quant: Dict[str, str] = field(default_factory=dict)
    hot_experts: int = 4
    # 检索预算（"知识不进上下文"的硬约束）
    kb_top_k: int = 3
    kb_max_tokens_per_segment: int = 64
    kb_max_total_tokens: int = 128
    # 路由
    router_confidence_low: float = 0.35
    router_confidence_high: float = 0.85
    general_expert: int = 0          # 常识兜底通路（低置信时只走它）
    aux_loss_weight: float = 0.01
    # 路由监督（oracle 蒸馏）权重：把"这个领域该走哪个专家"直接教给路由器。
    # 只靠交叉熵 + 负载均衡时，交叉熵一收敛，辅助损失就主导梯度并把路由按回均匀
    # （实测置信度 0.062 ≈ 1/16，专家等于被乘了个接近零的系数）。0 表示不监督。
    router_supervision_weight: float = 0.0

    def __post_init__(self):
        if isinstance(self.linear_attn, dict):
            self.linear_attn = LinearAttnConfig(**self.linear_attn)
        if isinstance(self.corrector, dict):
            self.corrector = CorrectorConfig(**self.corrector)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, "hidden 必须能被 head 数整除"
        assert self.num_attention_heads % self.num_kv_groups == 0, "head 数必须能被 KV group 整除"
        assert 1 <= self.top_k_experts <= self.num_skill_experts
        # 层号 1-indexed，与规格文档一致
        self._full = set(int(x) for x in self.full_attn_layers)
        assert all(1 <= x <= self.num_layers for x in self._full), "full_attn_layers 越界"

    def is_full_attention(self, layer_index_1based: int) -> bool:
        return layer_index_1based in self._full

    @property
    def num_full_layers(self) -> int:
        return len(self._full)

    @property
    def num_linear_layers(self) -> int:
        return self.num_layers - len(self._full)

    @classmethod
    def from_json(cls, path: str | Path) -> "KSRConfig":
        data: Dict[str, Any] = json.loads(Path(path).read_text("utf-8"))
        # 只有计算器用的字段，模型侧忽略（其余未知字段必须报错，防止拼错的规格被静默吞掉）
        for key in ("_note", "limits", "link", "runtime_overhead_bytes"):
            data.pop(key, None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise SystemExit(f"配置含未知字段 {sorted(unknown)}（拼错的规格会被静默忽略，所以这里必须报错）")
        return cls(**data)
