"""知识外挂：向量索引（持久化 embedding）、检索预算、专家绑定、纠正回写。

修掉原稿的两个致命问题：
  * 原稿 `_save_domain` 不写 embedding、`_load_domain` 读 `doc.get("embedding")`
    → 重启后全部是零向量，检索退化成随机。这里用 .npy 边车文件持久化，
    并有一条"写入→重载→检索结果一致"的测试守着。
  * 原稿的检索没有预算：片段数量、单片长度、总量都不受控 → KV Cache 照样涨。
    这里 retrieve() 强制受三重预算约束。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class KnowledgeRecord:
    rid: str
    domain: str
    kind: str                      # fact | rule | correction
    text: str                      # 注入上下文的正文
    answer: str = ""               # 该条知识对应的正确输出（用于自动判定是否答对）
    rule_text: str = ""            # 规则化表述，例如 "Windows 平台禁用 lsof"
    source: str = "manual"         # manual | correction | tool
    verified: bool = False
    created_at: str = field(default_factory=_now)
    meta: Dict = field(default_factory=dict)

    @staticmethod
    def make_id(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class VectorIndex:
    """单领域索引：jsonl 存正文 + .npy 存向量（行序一致）。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.records: List[KnowledgeRecord] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, rec: KnowledgeRecord, emb: Sequence[float]) -> None:
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        assert v.shape == (self.dim,), f"embedding 维度 {v.shape} != {self.dim}"
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        self.records.append(rec)
        self.vectors = np.concatenate([self.vectors, v[None]], axis=0)

    def search(self, q: Sequence[float], top_k: int) -> List[Tuple[float, KnowledgeRecord]]:
        if not self.records:
            return []
        qv = np.asarray(q, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(qv))
        if n == 0:
            return []
        sims = self.vectors @ (qv / n)
        order = np.argsort(-sims)[: max(0, top_k)]
        return [(float(sims[i]), self.records[i]) for i in order if self.records[i] is not None]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        np.save(path.with_suffix(".emb.npy"), self.vectors)   # ← 原稿丢的就是这个

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        vecs = np.load(path.with_suffix(".emb.npy"))
        idx = cls(dim=int(vecs.shape[1]))
        idx.vectors = vecs.astype(np.float32)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    idx.records.append(KnowledgeRecord.from_dict(json.loads(line)))
        assert len(idx.records) == idx.vectors.shape[0], "记录与向量行数不一致"
        return idx

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class Segment:
    score: float
    record: KnowledgeRecord
    text: str
    tokens: int


class KnowledgeBase:
    """SSD 上的多领域知识库。检索必须受预算约束——这是"低硬件"能否成立的关键。"""

    def __init__(self, root: str | Path, dim: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.indices: Dict[str, VectorIndex] = {}
        for p in sorted(self.root.glob("*.jsonl")):
            self.indices[p.stem] = VectorIndex.load(p)

    def _index(self, domain: str) -> VectorIndex:
        if domain not in self.indices:
            self.indices[domain] = VectorIndex(self.dim)
        return self.indices[domain]

    def add(self, domain: str, rec: KnowledgeRecord, emb: Sequence[float]) -> None:
        idx = self._index(domain)
        idx.add(rec, emb)
        idx.save(self.root / f"{domain}.jsonl")

    def size(self) -> int:
        return sum(len(v) for v in self.indices.values())

    def domains(self) -> List[str]:
        return sorted(self.indices)

    def retrieve(self, query_emb: Sequence[float], domains: Sequence[str], *, top_k: int,
                 max_tokens_per_segment: int, max_total_tokens: int,
                 count_tokens: Callable[[str], int],
                 query_text: str = "", lexical_weight: float = 1.0) -> List[Segment]:
        """按分数取 Top-K，逐段截断到单片上限，累计超过总预算就停。

        打分 = 向量相似度 + 词面关键片段命中。只用向量相似度不够：同一批问题共用一个
        句式模板，均值池化出来的嵌入几乎一样，留出集实测检索命中率只有 17%。
        生产环境的检索也都是稀疏+稠密混合——这里把"问题里的编号/ASCII 标识符有没有
        出现在条目里"作为一个硬信号加进来。

        返回的 token 总数一定 <= max_total_tokens（测试守着这条）。
        """
        qkeys = _key_tokens(query_text)
        # 先用向量分数**多召回**（候选池要明显大于 top_k），再用混合分数重排。
        # 只在 top_k 个向量近克里加词面信号是没用的：正确条目压根进不了前 k 名，
        # 权重加不到它头上——这就是上一版"加了混合检索但命中率纹丝不动"的原因。
        pool = max(top_k * 8, 32)
        cands: List[Tuple[float, KnowledgeRecord]] = []
        for d in domains:
            if d not in self.indices:
                continue
            for score, rec in self.indices[d].search(query_emb, pool):
                bonus = lexical_weight if (qkeys & _key_tokens(rec.text)) else 0.0
                cands.append((score + bonus, rec))
        cands.sort(key=lambda x: -x[0])

        out: List[Segment] = []
        used = 0
        for score, rec in cands:
            if len(out) >= top_k or used >= max_total_tokens:
                break
            text, n = _truncate_to_tokens(rec.text, max_tokens_per_segment, count_tokens)
            room = max_total_tokens - used
            if n > room:
                text, n = _truncate_to_tokens(rec.text, room, count_tokens)
            if n <= 0:
                continue
            used += n
            out.append(Segment(score=score, record=rec, text=text, tokens=n))
        return out


def _key_tokens(text: str) -> set:
    """抽出文本里真正能区分的片段：数字串和 ASCII 标识符。

    中文句式模板里的字对"是哪一条"没有区分度，有区分度的只有编号这类符号。
    """
    return set(re.findall(r"\d{2,}|[A-Za-z][A-Za-z0-9_.\-]{2,}", text))


def _truncate_to_tokens(text: str, max_tokens: int, count: Callable[[str], int]) -> Tuple[str, int]:
    """按 token 截断（不是按字符），用二分找边界。"""
    if max_tokens <= 0:
        return "", 0
    n = count(text)
    if n <= max_tokens:
        return text, n
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    return cut, max(count(cut), 1)


class ExpertBindings:
    """专家 → 知识库绑定。默认绑定 + 允许跨库补检索（写死不行的那一条）。"""

    def __init__(self, primary: Dict[int, List[str]], aux: Optional[Dict[int, List[str]]] = None):
        self.primary = {int(k): list(v) for k, v in primary.items()}
        self.aux = {int(k): list(v) for k, v in (aux or {}).items()}

    @classmethod
    def from_json(cls, path: str | Path) -> "ExpertBindings":
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(data.get("primary", {}), data.get("aux", {}))

    def domains_for(self, expert_id: int, include_aux: bool = False) -> List[str]:
        out = list(self.primary.get(expert_id, []))
        if include_aux:
            out += [d for d in self.aux.get(expert_id, []) if d not in out]
        return out


@dataclass
class CorrectionRecord:
    record_id: str
    query: str
    wrong_output: str
    corrected_output: str
    rule_text: str
    domain_tag: str
    error_type: str
    correction_source: str          # user | auto | tool
    status: str                     # pending | approved | rejected
    timestamp: str
    reviewer: str = ""
    hits: int = 0                   # 这条纠正后来被检索命中过几次

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CorrectionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CorrectionStore:
    """踩过的坑：pending → approved 才进知识库（避免把错的"纠正"当真理）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, status: str) -> Path:
        return self.root / f"{status}.jsonl"

    def submit(self, query: str, wrong: str, corrected: str, rule_text: str = "",
               domain_tag: str = "", error_type: str = "fact_mismatch",
               source: str = "user") -> CorrectionRecord:
        rid = hashlib.md5(f"{query}|||{wrong}".encode("utf-8")).hexdigest()[:12]
        rec = CorrectionRecord(record_id=rid, query=query, wrong_output=wrong,
                               corrected_output=corrected, rule_text=rule_text or corrected,
                               domain_tag=domain_tag, error_type=error_type,
                               correction_source=source, status="pending", timestamp=_now())
        with self._path("pending").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return rec

    def _move(self, rid: str, dst: str) -> Optional[CorrectionRecord]:
        src = self._path("pending")
        if not src.exists():
            return None
        keep, found = [], None
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["record_id"] == rid and found is None:
                found = CorrectionRecord.from_dict(d)
            else:
                keep.append(line)
        if found is None:
            return None
        src.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        found.status = dst
        found.timestamp = _now()
        with self._path(dst).open("a", encoding="utf-8") as f:
            f.write(json.dumps(found.to_dict(), ensure_ascii=False) + "\n")
        return found

    def approve(self, rid: str, reviewer: str = "admin") -> Optional[CorrectionRecord]:
        rec = self._move(rid, "approved")
        if rec:
            rec.reviewer = reviewer
        return rec

    def reject(self, rid: str, reviewer: str = "admin") -> Optional[CorrectionRecord]:
        return self._move(rid, "rejected")

    def load_approved(self, domain_tag: str = "") -> List[CorrectionRecord]:
        p = self._path("approved")
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = CorrectionRecord.from_dict(json.loads(line))
            if domain_tag and rec.domain_tag not in ("", domain_tag):
                continue
            out.append(rec)
        return out
