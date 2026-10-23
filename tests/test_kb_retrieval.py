"""知识库检索的回归测试（不需要 torch，只依赖 numpy）。

每条测试都对应一个真实测出来的缺陷：
  * 候选池写死 → 库从 64 条热插到 264 条，检索命中率 100%→62%，旧题准确率 71%→38%，
    而模型权重一个字节都没变。这不是模型的锅，是"正确条目进不了候选池"。
  * 只有稠密通路时，能不能命中取决于运气；符号（编号）精确命中不该靠碰运气。
  * 倒排不落盘，必须能从 records 重建，否则重启后稀疏通路静默失效。
  * 两条通路会把同一条目各送一次，不去重就会把检索预算占两遍。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ksr.knowledge import KnowledgeBase, KnowledgeRecord  # noqa: E402

DIM = 16
TOP_K = 2
MAX_SEG_TOKENS = 48
MAX_TOTAL_TOKENS = 96


def _count(text: str) -> int:
    return len(text)


def _fact(port: int, svc: str) -> str:
    return f"端口 {port} 的服务是 {svc}"


def _query(port: int) -> str:
    return f"查询端口 {port} 上运行的服务是什么\nA: "


def _kb(root: Path, n: int, base_port: int = 9000) -> KnowledgeBase:
    """建一个"句式完全相同、只有编号不同"的库——这正是最难的情况。

    向量刻意全部相同（全 1），稠密通路就彻底排不出名次：谁在前全看排序算法。
    这样还能 100% 命中，才说明命中是靠符号通路，不是靠稠密召回运气好。
    """
    kb = KnowledgeBase(root, dim=DIM)
    ones = np.ones(DIM, dtype=np.float32)
    for i in range(n):
        port, svc = base_port + i, f"svc{i % 24:02d}"
        text = _fact(port, svc)
        kb.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(text), domain="fact",
                                      kind="fact", text=text, answer=svc, source="manual",
                                      verified=True), ones)
    return kb


def _hit(kb: KnowledgeBase, port: int) -> bool:
    """检索一次，看正确条目是不是被送回来了。"""
    segs = kb.retrieve(np.ones(DIM, dtype=np.float32), kb.domains(), top_k=TOP_K,
                       max_tokens_per_segment=MAX_SEG_TOKENS,
                       max_total_tokens=MAX_TOTAL_TOKENS, count_tokens=_count,
                       query_text=_query(port))
    return any(s.record.text == _fact(port, f"svc{(port - 9000) % 24:02d}") for s in segs)


def test_symbol_recall_holds_when_dense_pool_is_capped():
    """库 2000 条 + 池子压到 32：稠密通路最多覆盖 1.6%，必须靠符号通路命中。"""
    with tempfile.TemporaryDirectory() as td:
        kb = _kb(Path(td), 2000)
        assert kb.size() == 2000
        cap = KnowledgeBase.SCAN_CAP
        try:
            KnowledgeBase.SCAN_CAP = 32          # 故意不让池子跟着库放大
            ports = [9000 + i for i in range(0, 2000, 7)]
            hits = sum(_hit(kb, p) for p in ports)
            assert hits == len(ports), f"符号通路失效：{hits}/{len(ports)} 命中"
            # 对照组：不给 query_text（等于没有符号信号）时，稠密通路应当基本抓不到。
            # 没有这个对照，上面那个 100% 可能只是运气好。
            blind = 0
            for p in ports[:60]:
                segs = kb.retrieve(np.ones(DIM, dtype=np.float32), kb.domains(), top_k=TOP_K,
                                   max_tokens_per_segment=MAX_SEG_TOKENS,
                                   max_total_tokens=MAX_TOTAL_TOKENS, count_tokens=_count,
                                   query_text="")
                blind += int(any(s.record.text == _fact(p, f"svc{(p - 9000) % 24:02d}")
                                 for s in segs))
            assert blind <= 6, f"稠密通路不该几乎全中（实际 {blind}/60），测试失去意义"
        finally:
            KnowledgeBase.SCAN_CAP = cap


def test_inverted_index_is_rebuilt_on_reload():
    """倒排不落盘，重启后必须从 records 重建——否则稀疏通路会静默失效。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        kb = _kb(root, 200)
        kb2 = KnowledgeBase(root, dim=DIM)
        assert kb2.size() == 200, f"重载后条目数不对：{kb2.size()}"
        assert kb2.indices["fact"].inverted, "重载后倒排是空的"
        ports = [9000 + i for i in range(0, 200, 11)]
        hits = sum(_hit(kb2, p) for p in ports)
        assert hits == len(ports), f"重载后符号通路失效：{hits}/{len(ports)}"


def test_merged_paths_do_not_inject_the_same_record_twice():
    """两条通路会命中同一条目：去重没做对的话，top_k=2 会被同一条占两遍。"""
    with tempfile.TemporaryDirectory() as td:
        kb = _kb(Path(td), 64)
        for p in (9000, 9003, 9011, 9027, 9050):
            segs = kb.retrieve(np.ones(DIM, dtype=np.float32), kb.domains(), top_k=TOP_K,
                               max_tokens_per_segment=MAX_SEG_TOKENS,
                               max_total_tokens=MAX_TOTAL_TOKENS, count_tokens=_count,
                               query_text=_query(p))
            rids = [s.record.rid for s in segs]
            assert len(rids) == len(set(rids)), f"同一条目被注入了两次：{rids}"


def test_token_budget_holds_with_two_paths_merged():
    """合并两条通路不能破坏三重预算：片数、单片长度、总量都得守住。"""
    with tempfile.TemporaryDirectory() as td:
        kb = _kb(Path(td), 300)
        for p in (9000, 9077, 9199):
            segs = kb.retrieve(np.ones(DIM, dtype=np.float32), kb.domains(), top_k=TOP_K,
                               max_tokens_per_segment=MAX_SEG_TOKENS,
                               max_total_tokens=MAX_TOTAL_TOKENS, count_tokens=_count,
                               query_text=_query(p))
            assert len(segs) <= TOP_K, f"片数超预算：{len(segs)}"
            assert all(s.tokens <= MAX_SEG_TOKENS for s in segs), "单片超预算"
            assert sum(s.tokens for s in segs) <= MAX_TOTAL_TOKENS, "总量超预算"


def test_remove_and_find_keep_records_and_vectors_aligned():
    """改一条知识要先能删掉旧的——只有 add 的话，"改"就变成库里新旧并存。

    对应 R4（重复权威）：库里同时有两条讲同一个端口，模型自己挑一条，更新就不生效。
    删中间一条最容易出事的是向量行号和记录错开，所以删完必须验证两者数量一致、
    剩下的每条都还能检索到。
    """
    def vec(n: int) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        v[n % DIM] = 1.0
        return v

    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(tmp, dim=DIM)
        for i, port in enumerate((8001, 8002, 8003)):
            text = f"端口 {port} 的服务是 svc{i}"
            kb.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(text), domain="fact",
                                          kind="fact", text=text, answer=f"svc{i}",
                                          source="manual", verified=True), vec(port))
        assert kb.size() == 3

        hits = kb.find("端口 8002 的服务是")
        assert len(hits) == 1 and hits[0][0] == "fact", hits
        assert kb.remove("fact", hits[0][1].rid) is True
        assert kb.remove("fact", "不存在的 rid") is False
        assert kb.size() == 2

        idx = kb.indices["fact"]
        assert len(idx.records) == idx.vectors.shape[0] == 2, "记录和向量行号错开了"
        for port in (8001, 8003):          # 剩下的两条必须都还能检索到
            segs = kb.retrieve(vec(port), ["fact"], top_k=TOP_K,
                               max_tokens_per_segment=MAX_SEG_TOKENS,
                               max_total_tokens=MAX_TOTAL_TOKENS, count_tokens=_count,
                               query_text=f"查询端口 {port} 的服务")
            assert segs and f"端口 {port}" in segs[0].text, (port, [s.text for s in segs])
            assert not any("8002" in s.text for s in segs), "删掉的条目还在被检索到"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
