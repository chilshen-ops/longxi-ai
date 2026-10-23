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
        # 符号倒排：编号/标识符 → 含它的行号。稠密召回受候选池限制，稀疏召回不受，
        # 所以库涨到几万条也不会把正确条目挤出候选。不落盘——从 records 就能重建，
        # 省掉一份会不同步的副本。
        self.inverted: Dict[str, List[int]] = {}

    def _rebuild_inverted(self) -> None:
        self.inverted = {}
        for row, rec in enumerate(self.records):
            for key in _key_tokens(rec.text):
                self.inverted.setdefault(key, []).append(row)

    def add(self, rec: KnowledgeRecord, emb: Sequence[float]) -> None:
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        assert v.shape == (self.dim,), f"embedding 维度 {v.shape} != {self.dim}"
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        # rid 是按正文哈希算的，所以"同一条知识"必然同 rid。重复 add 必须是覆盖而不是追加：
        # 实测过，演示台每跑一次就把整库重新灌一遍，64 条涨到 383 条，"改一条"一次要删掉两条。
        # 覆盖而不是跳过，是因为同一条正文可能带着更正后的 answer——那种情况要的是新值。
        for i, old in enumerate(self.records):
            if old.rid == rec.rid:
                self.records[i] = rec
                self.vectors[i] = v
                self._rebuild_inverted()      # 正文可能变了，倒排必须重建
                return
        self.records.append(rec)
        self.vectors = np.concatenate([self.vectors, v[None]], axis=0)
        row = len(self.records) - 1
        for key in _key_tokens(rec.text):
            self.inverted.setdefault(key, []).append(row)

    def remove(self, rid: str) -> bool:
        """按 rid 删一条，并重建符号倒排。

        为什么要有：改一条知识必须先能撤掉旧的。curator 的否决、R4 说的"重复权威"消歧，
        都要求库里对同一个事实最多只有一条——否则模型会自己挑一条，更新就不生效。
        实现上整表重灌（add 会顺手把倒排建好、向量再归一一次是无害的），这个规模无所谓。
        """
        if not any(r.rid == rid for r in self.records):
            return False
        keep = [(r, v) for r, v in zip(self.records, self.vectors) if r.rid != rid]
        self.records = []
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.inverted = {}
        for rec, vec in keep:
            self.add(rec, vec)
        return True

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

    def search_keys(self, qkeys: set, limit: int) -> List[Tuple[float, KnowledgeRecord]]:
        """符号通路：查询里出现的编号，直接查到含它的条目。

        不看相似度、也不受候选池大小影响，所以库涨到几万条命中率也不会掉。
        命中几个符号就算几分，和词面加分同一量纲，交给同一个重排去比。
        """
        if not qkeys or limit <= 0:
            return []
        hits: Dict[int, int] = {}
        for key in qkeys:
            for row in self.inverted.get(key, ()):
                hits[row] = hits.get(row, 0) + 1
        ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(float(n), self.records[row]) for row, n in ranked]

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
        for row, rec in enumerate(idx.records):          # 倒排从 records 重建
            for key in _key_tokens(rec.text):
                idx.inverted.setdefault(key, []).append(row)
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

    # 暴力全库扫的召回上限。超过这个规模就不该再靠扫全库兜底，必须换 faiss 或符号倒排。
    SCAN_CAP = 4096

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

    def remove(self, domain: str, rid: str) -> bool:
        """删一条并落盘（jsonl 和向量侧车一起重写，行序保持一致）。"""
        idx = self.indices.get(domain)
        if idx is None or not idx.remove(rid):
            return False
        idx.save(self.root / f"{domain}.jsonl")
        return True

    def find(self, needle: str, domain: str = "") -> List[Tuple[str, KnowledgeRecord]]:
        """按正文包含关系定位条目，返回 (领域, 记录)。给"改一条知识"找 rid 用。"""
        domains = [domain] if domain else self.domains()
        hits: List[Tuple[str, KnowledgeRecord]] = []
        for dom in domains:
            idx = self.indices.get(dom)
            if idx is None:
                continue
            hits.extend((dom, r) for r in idx.records if needle in r.text)
        return hits

    def size(self) -> int:
        return sum(len(v) for v in self.indices.values())

    def domains(self) -> List[str]:
        return sorted(self.indices)

    def retrieve(self, query_emb: Sequence[float], domains: Sequence[str], *, top_k: int,
                 max_tokens_per_segment: int, max_total_tokens: int,
                 count_tokens: Callable[[str], int],
                 query_text: str = "", lexical_weight: float = 1.0) -> List[Segment]:
        """按分数取 Top-K，逐段截断到单片上限，累计超过总预算就停。

        召回有**两条通路，缺一不可**（合并去重后统一重排）：
          * 稠密：向量相似度先多召回再重排。只用向量相似度不够——同一批问题共用句式模板，
            均值池化出来的嵌入几乎一样，留出集实测只有 17%。
          * 符号：`_key_tokens` 抽出的编号/标识符走倒排精确命中，**与库规模无关**。
            只有稠密通路时，正确条目能不能进候选池要看运气：热插 200 条把库从 64 涨到
            264，命中率就从 100% 掉到 62%、旧题准确率 71%→38%——而模型权重一个字节没变。
        排序是**两级**的，不是把两路分数加在一起：先按符号命中数降序，再按混合分降序。
        这条是被测出来的：把"编号精确命中"设计成和 cosine 同量纲的 +1.0 加分，只要条目
        向量彼此接近（同句式模板就是这样，cosine≈1），精确命中就会和纯相似**打平**，
        打平就退化成按插入顺序排——符号通路等于白做（2000 条库、池子压到 32 时实测只有
        5/286 命中）。精确编号命中本来就该压过任何相似度。

        返回的 token 总数一定 <= max_total_tokens（测试守着这条）。
        """
        qkeys = _key_tokens(query_text)
        # 稠密通路的候选池必须随库规模放大，不能写死。search() 本来就要对整个索引算一遍
        # 相似度再排序，截断只是把已经算出来的结果扔掉，所以放大池子几乎不增加成本
        # （实测单条更新 2.3ms→2.7ms）。超过 SCAN_CAP 就不该再靠扫全库，交给 faiss。
        scope = [d for d in domains if d in self.indices]
        total = sum(len(self.indices[d]) for d in scope)
        pool = max(top_k * 8, 32, min(total, self.SCAN_CAP))
        # rid -> (混合分, 符号命中数, 记录)。两条通路会把同一条目各送一次，不去重就会把
        # 检索预算占两遍；同一个 rid 取 (符号数, 混合分) 较高的那次。
        best: Dict[str, Tuple[float, int, KnowledgeRecord]] = {}

        def offer(score: float, rec: KnowledgeRecord, sym: int) -> None:
            prev = best.get(rec.rid)
            if prev is None or (sym, score) > (prev[1], prev[0]):
                best[rec.rid] = (score, sym, rec)

        for d in scope:
            idx = self.indices[d]
            for score, rec in idx.search(query_emb, pool):
                keys = qkeys & _key_tokens(rec.text)
                offer(score + (lexical_weight if keys else 0.0), rec, len(keys))
            for n, rec in idx.search_keys(qkeys, pool):
                offer(n * lexical_weight, rec, n)
        cands = sorted(best.values(), key=lambda v: (-v[1], -v[0]))

        out: List[Segment] = []
        used = 0
        for score, _sym, rec in cands:
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
