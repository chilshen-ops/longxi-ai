r"""分开测两条路：模型自己调工具去查 vs 回答前自动检索。

为什么要分开：聊天台现在两条路都在跑，答对了不知道是谁的功劳，答错了也不知道该怪谁。
这个基准把自动检索关掉，只留工具，量一次；再打开，量一次。三件事就分开了：

  该查的时候它自己去查了吗        —— 这是模型的意愿问题，只能换更大的模型
  该写的时候它自己去写了吗        —— 同上
  检索本身准不准、拿到之后它认不认账 —— 这是检索和提示的问题，能改

不该动工具的时候它动了没有，也量出来——那是误触发，比不查更烦人。

用法：
    .venv-gpu\Scripts\python.exe tools/chatbench.py            # 两种模式都跑
    .venv-gpu\Scripts\python.exe tools/chatbench.py --mode tool # 只跑工具路径
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import chatui as C  # noqa: E402

# 每条：编号、类型、要先塞进知识库的事实、要问的话、期望它调什么、答案里必须出现什么。
# 事实都选"模型不可能预先知道"的私人事实——不然它凭参数就能答对，测不出有没有去查。
CASES = [
    {"id": "R1", "kind": "recall", "seed": ["我的工号是 483920"],
     "ask": "我的工号是多少？", "want": "483920"},
    {"id": "R2", "kind": "recall", "seed": ["我们项目的部署机器是 10.14.7.22"],
     "ask": "我们项目的部署机器是哪台？", "want": "10.14.7.22"},
    {"id": "R3", "kind": "recall", "seed": ["我女儿的生日是 3 月 12 日"],
     "ask": "我女儿的生日是哪天？", "want": "3 月 12"},
    {"id": "R4", "kind": "recall", "seed": ["生产数据库的连接串存放在 vault 的 secret/db/prod 路径下"],
     "ask": "生产数据库的连接串存放在哪里？", "want": "secret/db/prod"},
    {"id": "R5", "kind": "recall", "seed": ["我们内部把灰度发布系统叫作灯塔"],
     "ask": "我们的灰度发布系统内部叫什么名字？", "want": "灯塔"},
    {"id": "R6", "kind": "recall", "seed": ["王工负责客服系统，李工负责计费系统"],
     "ask": "客服系统是谁在负责？", "want": "王工"},
    {"id": "W1", "kind": "write", "seed": [],
     "ask": "记一下，我家猫的生日是 8 月 3 日", "want": "kb_add"},
    {"id": "W2", "kind": "write", "seed": [],
     "ask": "请记住：我们的测试环境密码是 Kx7-2213", "want": "kb_add"},
    {"id": "W3", "kind": "write", "seed": [],
     "ask": "帮我记住，周报每周五下午四点之前交", "want": "kb_add"},
    {"id": "N1", "kind": "none", "seed": [], "ask": "用一句话解释什么是 HTTP", "want": ""},
    {"id": "N2", "kind": "none", "seed": [], "ask": "你是谁？", "want": ""},
    {"id": "N3", "kind": "none", "seed": [], "ask": "帮我把这句话改通顺：因为所以科学道理",
     "want": ""},
]

from ksr.knowledge import KnowledgeBase  # noqa: E402


def reset_kb(seeds: list[str]) -> None:
    """每条用例都从干净的库开始，再把这条要的事实直接写进去。

    种子事实是用 tool_kb_add 写、不是让模型写： seeding 里不该掺进模型的不稳定，
    否则测出来的"它不去查"可能只是"它上次没写进去"。
    """
    for stale in Path(C.KB_DIR).glob("chat*"):
        stale.unlink()
    C.CTX["kb"] = KnowledgeBase(C.KB_DIR, dim=C.CTX["dim"])
    for text in seeds:
        C.tool_kb_add({"text": text})


def run(mode: str, only: str = "") -> dict:
    """mode=tool 只许模型自己调工具（自动检索关掉）；mode=auto 回答前先自动检索。"""
    s = {"recall_called": 0, "recall_right": 0, "write_called": 0, "false_fire": 0,
         "retrieve_hit": 0, "n": {"recall": 0, "write": 0, "none": 0}, "lines": []}
    for case in CASES:
        if only and case["kind"] != only:
            continue
        reset_kb(case["seed"])
        r = C.chat([{"role": "user", "content": case["ask"]}], auto=(mode == "auto"))
        tools = [st["tool"] for st in r["steps"]]
        reply = r["reply"] or ""
        s["n"][case["kind"]] += 1
        if case["kind"] == "recall":
            if "kb_lookup" in tools:
                s["recall_called"] += 1
            if case["want"] in reply:
                s["recall_right"] += 1
            got = [x["text"] for st in r["steps"] if st["tool"] == "auto_retrieve"
                   for x in (st["result"].get("results") or [])]
            if any(case["want"] in t for t in got):
                s["retrieve_hit"] += 1
        elif case["kind"] == "write":
            if "kb_add" in tools:
                s["write_called"] += 1
        elif tools:
            s["false_fire"] += 1
        s["lines"].append(
            f"  {case['id']} {case['kind']:<7} 调了 {','.join(tools) or '（没调）':<22} "
            f"答：{reply[:44]}")
        print(s["lines"][-1], flush=True)
    return s


def report(all_stats: dict) -> None:
    print("\n" + "=" * 72)
    for mode, s in all_stats.items():
        nr, nw, nn = (s["n"]["recall"], s["n"]["write"], s["n"]["none"])
        print(f"\n【{mode}】")
        print(f"  该查的时候自己去查了   {s['recall_called']}/{nr}")
        print(f"  该查的时候答对了       {s['recall_right']}/{nr}")
        if mode == "auto":
            print(f"  检索命中了正确事实     {s['retrieve_hit']}/{nr}")
        print(f"  该写的时候自己去写了   {s['write_called']}/{nw}")
        print(f"  不该动工具却动了       {s['false_fire']}/{nn}")
    print("\n读法：tool 那一列低，说明该换模型；auto 的「答对了」高但「检索命中」低，"
          "说明该修检索；两个都高，才是真的能用。")


def main() -> int:
    ap = argparse.ArgumentParser(description="分开测工具路径和自动检索路径")
    ap.add_argument("--mode", choices=["tool", "auto", "both"], default="both")
    ap.add_argument("--only", choices=["recall", "write", "none", ""], default="",
                    help="只跑某一类用例")
    args = ap.parse_args()
    C.boot()
    if not C.CTX["ready"]:
        print("加载失败：", C.CTX["error"])
        return 1
    print(f"模型 {C.CTX['params']:,} 参数；{len(CASES)} 条用例；"
          f"模式 {'tool + auto' if args.mode == 'both' else args.mode}\n")
    modes = ["tool", "auto"] if args.mode == "both" else [args.mode]
    all_stats = {}
    for m in modes:
        print(f"\n{'#' * 24} {m} {'#' * 24}")
        all_stats[m] = run(m, args.only)
    report(all_stats)
    (ROOT / "work-chat" / "bench.json").parent.mkdir(exist_ok=True)
    (ROOT / "work-chat" / "bench.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
