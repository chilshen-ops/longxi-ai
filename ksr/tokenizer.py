"""字符级 tokenizer：词表由语料构建，未见字符走 UTF-8 字节回退（不会丢信息）。

与"随机词表 + UNK"的做法不同，这里保证 encode/decode 是无损往返：
任何字符串 encode 后再 decode 都还原原文（特殊标记除外）。这是知识库回写
链路能被测试覆盖的前提——如果编码有损，"命中已纠正案例"就无法判定。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

PAD, BOS, EOS, SEP = "<PAD>", "<BOS>", "<EOS>", "<SEP>"
SPECIALS = [PAD, BOS, EOS, SEP]
_BYTE_PREFIX = "<B%02X>"


class CharTokenizer:
    """字符级 tokenizer + 字节回退。词表小、确定性强，适合原型验证。"""

    def __init__(self, vocab: Iterable[str] = (), extra_tokens: Iterable[str] = ()):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[str, str] = {}
        for t in SPECIALS:
            self._add(t)
        # 字节回退槽位固定放在特殊 token 之后，保证跨语料稳定
        for i in range(256):
            self._add(_BYTE_PREFIX % i)
        for t in vocab:
            self._add(t)
        for t in extra_tokens:
            self._add(t)
        self.pad_id = self.token2id[PAD]
        self.bos_id = self.token2id[BOS]
        self.eos_id = self.token2id[EOS]
        self.sep_id = self.token2id[SEP]

    # ── 词表 ────────────────────────────────────────────────────────────
    def _add(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)
            self.id2token[len(self.token2id) - 1] = token
        return self.token2id[token]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def from_corpus(cls, texts: Iterable[str]) -> "CharTokenizer":
        """把语料里出现过的所有字符收进词表。"""
        seen: List[str] = []
        seen_set = set()
        for text in texts:
            for ch in text:
                if ch not in seen_set:
                    seen_set.add(ch)
                    seen.append(ch)
        return cls(vocab=seen)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": list(self.id2token[i]) for i in range(self.vocab_size)},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(vocab=data["tokens"][len(SPECIALS) + 256:])

    # ── 编解码 ──────────────────────────────────────────────────────────
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids: List[int] = []
        if add_special:
            ids.append(self.bos_id)
        for ch in text:
            tok = self.token2id.get(ch)
            if tok is None:  # 字节回退：任何字符都能表示
                for byte in ch.encode("utf-8"):
                    ids.append(self.token2id[_BYTE_PREFIX % byte])
            else:
                ids.append(tok)
        if add_special:
            ids.append(self.eos_id)
        return ids

    def encode_query(self, text: str) -> List[int]:
        return self.encode(text, add_special=False)

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out: List[str] = []
        pending: List[int] = []  # 攒字节，遇到非字节 token 时统一解码

        def flush():
            if pending:
                raw = bytes(pending)
                out.append(raw.decode("utf-8", errors="replace"))
                pending.clear()

        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        for i in ids:
            i = int(i)
            if i in special_ids:
                flush()
                if not skip_special:
                    out.append(self.id2token[i])
                continue
            tok = self.id2token.get(i, "")
            if tok.startswith("<B") and len(tok) == 4:
                pending.append(int(tok[2:4], 16))
                continue
            flush()
            out.append(tok)
        flush()
        return "".join(out)

    def count_tokens(self, text: str) -> int:
        """检索预算用：按 token 计数，不是按字符。"""
        return len(self.encode(text, add_special=False))
