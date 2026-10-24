"""损失区间框选（tools/finetune.py 的 encode()）的回归测试。

这个 bug 不会报错，只会静默训坏：
  * 原来用「分别渲染前缀、拿 token 数相减」框区间，Qwen3 模板给最后一轮
    assistant 插了个空 think 块，两边一减区间整体错位。
  * 改成按字符定位之后，一度框过了头——把下一轮的用户消息和工具返回
    也框进损失。模型于是学会自己生成用户的话，训出来的 adapter 什么也
    干不成。

所以这里盯的是**框出来的区间必须恰好等于「正文 + <|im_end|>」**：
多框一个字符都不行，因为再往后就是「别人说的话」。

依赖 transformers 和一个 Qwen3 的 tokenizer（模型在仓库外）。
两者缺一个就跳过并说明原因，不算失败——这台机器之外的仓库副本
也该能跑这套测试。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

END_MARK = "<|im_end|>"


def _find_tokenizer():
    """找一个能用的 Qwen3 tokenizer。找不到就返回 None。"""
    cands = []
    env = os.environ.get("KSR_CHAT_MODEL", "").strip()
    if env:
        cands.append(Path(env))
    cands += [Path(r"A:\default-workspace\qwen3-1.7b"),
              Path(r"A:\default-workspace\qwen3-0.6b"),
              ROOT.parent / "qwen3-1.7b", ROOT.parent / "qwen3-0.6b"]
    for c in cands:
        if (c / "tokenizer.json").exists():
            return c
    return None


def _load():
    """返回 (tok, finetune, chatui)，依赖不齐就返回 None。"""
    mp = _find_tokenizer()
    if mp is None:
        return None
    try:
        from transformers import AutoTokenizer
        import chatui as C
        import finetune as F
    except ImportError as exc:      # 没装 transformers/peft 就不跑
        print(f"跳过：缺依赖 {exc}")
        return None
    return AutoTokenizer.from_pretrained(str(mp)), F, C


_LOADED = _load()


def _sample_msgs(C):
    """几条形状不同的对话，涵盖工具调用轮、纯回答轮、正文重复出现的情况。"""
    call = ('<tool_call>\n{"name": "kb_lookup", "arguments": {"query": "我的工号是多少？"}}\n'
            '</tool_call>')
    obs = '<tool_response>\n{"tool": "kb_lookup", "result": {"ok": true}}\n'
    return [
        # 一轮工具调用 + 一轮回答（最常见的形状）
        [{"role": "system", "content": C.SYSTEM},
         {"role": "user", "content": "我的工号是多少？"},
         {"role": "assistant", "content": call},
         {"role": "user", "content": obs},
         {"role": "assistant", "content": "你的工号是 483920。"}],
        # 只有一轮纯回答
        [{"role": "system", "content": C.SYSTEM},
         {"role": "user", "content": "用一句话解释什么是 HTTP"},
         {"role": "assistant", "content": "HTTP 是应用层协议。"}],
        # 同一段字在用户的话里也出现过——框错位置很容易
        [{"role": "system", "content": C.SYSTEM},
         {"role": "user", "content": "改一下这句：部署机器是 10.20.30.40"},
         {"role": "assistant", "content": "部署机器是 10.20.30.40"}],
        # 连续两轮 assistant（中间没有用户消息）
        [{"role": "system", "content": C.SYSTEM},
         {"role": "user", "content": "帮我算 12 * 13"},
         {"role": "assistant", "content": "12 * 13 = 156。"},
         {"role": "assistant", "content": "还有别的要算的吗？"}],
    ]


def _framed(tok, F, ids, labels):
    """从 labels 里取出所有被框住的段（解码成文本）。"""
    segs, i = [], 0
    while i < len(labels):
        if labels[i] != F.IGNORE:
            j = i
            while j < len(labels) and labels[j] != F.IGNORE:
                j += 1
            segs.append(tok.decode(ids[i:j]))
            i = j
        else:
            i += 1
    return segs


def test_framed_region_is_exactly_content_plus_end_mark():
    """每一轮 assistant：框出来的必须恰好是「正文 + <|im_end|>」。"""
    tok, F, C = _LOADED
    for msgs in _sample_msgs(C):
        ids, labels = F.encode(tok, msgs, C.TOOLS)
        assert ids is not None, "样本被丢了，这本身就不对"
        want = [m["content"] + END_MARK for m in msgs if m["role"] == "assistant"]
        got = _framed(tok, F, ids, labels)
        assert got == want, f"框出来的区间不对\n  期望 {want}\n  实际 {got}"


def test_loss_never_contains_other_peoples_words():
    """损失里不能出现下一轮的用户消息或工具返回——那些模型没生成过。

    只查结构性标记：assistant 的正文里本来就可能引用用户说过的话
    （工具调用带 query、改写类任务原样返回那句话），那是它自己该生成的，
    不能算"别人的话"。
    """
    tok, F, C = _LOADED
    for msgs in _sample_msgs(C):
        ids, labels = F.encode(tok, msgs, C.TOOLS)
        assert ids is not None
        for seg in _framed(tok, F, ids, labels):
            assert "<|im_start|>" not in seg, f"多框了下一轮的开头：{seg!r}"
            assert "<tool_response>" not in seg, f"多框了工具返回：{seg!r}"


def test_non_assistant_tokens_are_ignored():
    """除 assistant 正文外，其余 token 一律 IGNORE。"""
    tok, F, C = _LOADED
    msgs = _sample_msgs(C)[0]
    ids, labels = F.encode(tok, msgs, C.TOOLS)
    assert ids is not None
    n_ignored = sum(1 for x in labels if x == F.IGNORE)
    assert n_ignored > 0, "整段都算进损失了，那等于没做区间框选"
    # 忽略掉的必须占多数：prompt 通常比回答长得多
    assert n_ignored > len(labels) // 2, f"忽略的太少：{n_ignored}/{len(labels)}"


def test_unrenderable_row_is_dropped():
    """正文在渲染结果里对不上时，整条丢掉而不是框错位置。

    模板给最后一轮 assistant 渲染时会 lstrip 掉正文开头的换行，
    所以正文带前导换行时渲染结果里根本没有它——这种行必须丢掉。
    """
    tok, F, C = _LOADED
    msgs = [{"role": "system", "content": C.SYSTEM},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "\n这句话渲染不出来"}]
    ids, labels = F.encode(tok, msgs, C.TOOLS)
    assert ids is None and labels is None, "渲染不出来的正文却没丢掉这一条"


if __name__ == "__main__":
    if _LOADED is None:
        print("跳过：没有可用的 Qwen3 tokenizer（设 KSR_CHAT_MODEL 指向模型目录）")
        raise SystemExit(0)
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
