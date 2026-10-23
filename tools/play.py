"""KSR 演示台：往知识库里加一条事实，不重训，模型立刻能答对。

这是本项目**唯一有硬证据**的能力（见 docs/KNOWLEDGE.md 的 B3/B8），所以做成能敲命令的东西，
而不是让人去读评测脚本。它同时会如实显示这个模型**不**具备的东西：路由选专家——实测把全部
专家清零，答案一题都不变。

用法：
    python tools/play.py --work work-conflict
    python tools/play.py --work work-conflict --ask 9005
    python tools/play.py --work work-conflict --demo      # 自动跑一遍：问→加→立刻答对

命令：
    问 <端口> / 直接敲问题      查知识库并回答，显示检索到了什么、模型有没有采纳
    加 <端口> <服务名>          写入知识库（计时 + 证明权重指纹没变），然后立刻回答它
    改 <端口> <服务名>          先删旧条目再写新的——"哪里不对修哪里"
    删 <端口>                  把这条知识从库里去掉
    知识                        知识库规模与最近几条
    帮助 / 退出
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

if "--work" in sys.argv:
    os.environ["KSR_WORK"] = sys.argv[sys.argv.index("--work") + 1]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from ksr.config import KSRConfig  # noqa: E402
from ksr.knowledge import KnowledgeRecord  # noqa: E402
from ksr.nets import KSRModel  # noqa: E402
import ksr.experiment as E  # noqa: E402

from attrib import resolve_cfg_path  # noqa: E402

HELP = """命令：
  问 <端口>            查知识库并回答（也可以直接把问题打出来）
  加 <端口> <服务名>   写入知识库 → 立刻回答，全程不重训、不产生梯度
  改 <端口> <服务名>   先删掉旧条目再写新的（哪里不对修哪里）
  删 <端口>          把这条知识从库里去掉
  知识                 知识库规模与最近几条
  退出
"""


def load():
    """按检查点里记的配置重建模型，再挂上它训练时用的那个知识库。"""
    work = Path(os.environ["KSR_WORK"])
    state = torch.load(work / "model.pt", map_location="cpu", weights_only=False)
    cfg = KSRConfig.from_json(resolve_cfg_path(state))
    corpus = E.build_corpus()
    tok = E.build_tokenizer(corpus)
    E._TOK = tok
    cfg.vocab_size = tok.vocab_size
    model = KSRModel(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    embed = E.make_embed(model)
    kb = E.build_kb(model, tok, embed, corpus, cfg)
    return model, tok, embed, kb, corpus, cfg


def token_ok(tok, text: str) -> bool:
    """这个词是不是模型词表里的一个整 token。不是的话模型基本答不出来。"""
    return len(tok.encode(text)) == 1


def ask(model, tok, embed, kb, cfg, port: int, note: str = "") -> None:
    """完整展示一次带检索的回答：检索到什么、注入了几个 token、模型采不采纳。"""
    q = E.fact_query(port)
    segs = E.retrieve_for(model, tok, kb, embed, q, cfg)
    ctx = E.render_context(segs)
    prior, _ = E.answer_of(model, tok, q)
    ans, _ = E.answer_of(model, tok, ctx + q)
    hit = next((s.record.answer for s in segs if s.record.answer), "")
    used = "采纳了知识库" if ans == hit else ("坚持自己的先验" if ans == prior else "给了第三个答案")
    print(f"\n问题：{q}")
    if segs:
        print(f"  检索：命中 {len(segs)} 条，注入 {sum(s.tokens for s in segs)} token"
              f"（预算 {cfg.kb_max_total_tokens}）")
        for s in segs:
            print(f"    {s.text}   [混合分 {s.score:.3f}，含符号命中加成]")
    else:
        print("  检索：知识库里没有这条")
    print(f"  不检索时模型答：{prior}")
    print(f"  带检索时模型答：{ans}   ← {used}")
    if note:
        print(f"  {note}")


def add_fact(model, kb, embed, port: int, svc: str, replace: bool) -> None:
    """写一条知识：先删同题的旧条目（改的情况），再写入，并证明权重一个字节都没动。"""
    text = f"端口 {port} 的服务是 {svc}"
    fp_before = E.model_fingerprint(model)
    verb = "改" if replace else "加"
    for dom, old in ([] if not replace else kb.find(f"端口 {port} 的服务是")):
        kb.remove(dom, old.rid)
        print(f"  删掉旧条目：{old.text}")
    t0 = time.perf_counter()
    kb.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(text), domain="fact",
                                   kind="fact", text=text, answer=svc,
                                   source="manual", verified=True), embed(text))
    ms = (time.perf_counter() - t0) * 1000
    fp_after = E.model_fingerprint(model)
    print(f"  {verb}成功：{text}   用时 {ms:.1f} ms，知识库现有 {kb.size()} 条")
    print(f"  权重指纹 {fp_before[:16]}… → {fp_after[:16]}… "
          f"{'没变（一个梯度都没走）✓' if fp_before == fp_after else '变了 ✗'}")


def main() -> int:
    torch.set_num_threads(int(os.environ.get("KSR_THREADS", "2")))
    model, tok, embed, kb, corpus, cfg = load()
    print(f"模型：{sum(p.numel() for p in model.parameters()):,} 参数，"
          f"合成任务（端口→服务）；知识库 {kb.size()} 条，存在 {os.environ['KSR_WORK']}/knowledge")
    print("这个演示台展示的是已验证的能力：加一条事实，不重训就能答对。")
    print("它不展示路由选专家——实测把全部专家清零，答案一题都不变（docs/KNOWLEDGE.md）。")

    if "--ask" in sys.argv:
        port = int(re.search(r"\d+", sys.argv[sys.argv.index("--ask") + 1]).group())
        ask(model, tok, embed, kb, cfg, port)
        return 0
    if "--demo" in sys.argv:
        new_port = 9100
        # 服务名要选模型词表里的整 token：知识库能补"哪个端口对应哪个服务"，
        # 补不了"模型从没学过怎么读这个词"——这是这条能力的边界，演示里也要说清楚。
        new_svc = next((s for s in E.SERVICES
                        if token_ok(tok, s) and corpus["fact_table"].get(new_port) != s), "svc0")
        print(f"\n【1】先问一个知识库里没有的端口 {new_port}")
        ask(model, tok, embed, kb, cfg, new_port)
        print(f"\n【2】把「端口 {new_port} 的服务是 {new_svc}」写进知识库"
              f"（{new_svc} 是模型认识的词，但从没和这个端口配过对）")
        add_fact(model, kb, embed, new_port, new_svc, replace=True)   # replace=True：重复跑不会堆重复条目
        print(f"\n【3】同一个问题，再问一次（模型没有重新训练）")
        ask(model, tok, embed, kb, cfg, new_port,
            note="这就是『更新知识库不需要重训』：写入 2.6 ms 量级，权重指纹不变")
        return 0

    print()
    print(HELP)
    while True:
        try:
            line = input("ksr> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        low = line.lower()
        if low in ("退出", "exit", "quit", "q"):
            return 0
        if low in ("帮助", "help", "?", "h"):
            print(HELP)
            continue
        if low in ("知识", "kb"):
            print(f"知识库 {kb.size()} 条，领域：{kb.domains()}")
            for dom in kb.domains():
                for rec in kb.indices[dom].records[-3:]:
                    print(f"  [{dom}] {rec.text}   (来源 {rec.source}，已审核 {rec.verified})")
            continue
        m = re.match(r"^(加|改)\s+(\d{3,7})\s+(\S+)$", line)
        if m:
            port, svc = int(m.group(2)), m.group(3)
            if not token_ok(tok, svc):
                print(f"  注意：「{svc}」不在模型词表里（会被拆成 {len(tok.encode(svc))} 个字符），"
                      f"模型读不出来。知识库能补『哪个端口对应哪个服务』，"
                      f"补不了『模型从没学过怎么读这个词』。")
            elif svc not in set(corpus["fact_table"].values()):
                # 实测到的真实边界，不是猜的：「Panov」是词表里的一个整 token（另一个领域的词表里
                # 就有人名），检索也命中了（混合分 1.93），模型照样不抄。原因不是读不出词，
                # 而是它只在见过的答案形状（svcNN）里学会照抄上下文——认账是格式绑定的。
                print(f"  注意：「{svc}」模型**认识**（是一个整 token），但这个槽位上它从没被要求"
                      f"抄过这种形状的答案。接下来大概率不采纳——这不是读不出词，"
                      f"而是**认账只在它见过的答案形状里成立**。")
            add_fact(model, kb, embed, port, svc, replace=m.group(1) == "改")
            ask(model, tok, embed, kb, cfg, port)
            continue
        m = re.match(r"^删\s+(\d{3,7})\s*$", line)
        if m:
            port = int(m.group(1))
            gone = 0
            for dom, old in kb.find(f"端口 {port} 的服务是"):
                kb.remove(dom, old.rid)
                gone += 1
                print(f"  删掉：{old.text}")
            if not gone:
                print(f"  没有找到端口 {port} 的条目")
            ask(model, tok, embed, kb, cfg, port)
            continue
        m = re.match(r"^(问)?\s*(\d{3,7})\s*$", line)
        port = int(m.group(2)) if m else None
        if port is None:
            m2 = re.search(r"(\d{3,7})", line)
            port = int(m2.group(1)) if m2 else None
        if port is None:
            print("没看懂。输入 帮助 看命令。")
            continue
        ask(model, tok, embed, kb, cfg, port)


if __name__ == "__main__":
    raise SystemExit(main())
