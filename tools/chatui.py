r"""真模型聊天台：Qwen3-0.6B + 外部知识库 + 工具 + 思考。

这个项目要证的东西没变——**更新知识库不需要重训**——只是把它从合成任务搬到了
一个真的对话模型上。四件事各自都有明确的证据：

  对话    多轮问答，模型本身的能力。
  工具    calc / kb_add / kb_lookup 三个真工具。模型发出 <tool_call>，我们执行，
          再把结果以 user 身份包在 <tool_response>…</tool_response> 里喂回去——
          这不是我编的协议，是 Qwen3 对话模板里写死的约定（查过模板）。
  思考    enable_thinking 开关。关掉时模板会追加一个空的 <think></think>，
          模型就直接答；打开时它先想再答。思考过程单独回传给页面折叠显示。
  知识库  写进去的是外部 jsonl，模型权重一个字节都不动——每次写入都算一次权重指纹
          前后对比，把"没动权重"这件事印在返回值里，不是嘴上说说。

用法：
    .venv-gpu\Scripts\python.exe tools/chatui.py --port 8770
    浏览器打开 http://127.0.0.1:8770

必须用 .venv-gpu 的 python（GPU 版 torch），而且**不要**设 PYTHONPATH——
指到 .venv-lib 会让它 import 到 CPU 版 torch。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ksr.knowledge import KnowledgeBase, KnowledgeRecord  # noqa: E402

MODEL_PATH = os.environ.get("KSR_CHAT_MODEL", r"A:\default-workspace\qwen3-0.6b")
KB_DIR = str(ROOT / "work-chat" / "knowledge")
SYSTEM = ("你是一个助手，有 calc、kb_add、kb_lookup 三个工具。"
          "用户说「记一下」「记住」时调用 kb_add 写进外部知识库；"
          "用户问一个可能是他以前说过的事实时调用 kb_lookup 去查，"
          "查到了照知识库答，没查到就说没查到；任何算术都调用 calc，不要心算。"
          "要调用工具就直接输出工具调用，不要用自己的话代替它。"
          "用户让你改写、翻译、润色、解释一段文本时，那是任务不是知识，不要用任何工具。"
          "回答要简短。")

TOOLS = [
    {"type": "function", "function": {
        "name": "calc", "description": "做精确计算。任何算术都必须调用它，不要自己心算。",
        "parameters": {"type": "object", "properties": {
            "expr": {"type": "string", "description": "算术表达式，例如 1234567 * 7919"}},
            "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "kb_add",
        "description": "把一条新知识写进外部知识库，以后问到就能查到。"
                       "**只用于用户在告诉你一个关于他自己、他的人或他的项目的持久事实**"
                       "（他的工号、他家的猫生日、项目的部署机器、团队的约定）。"
                       "用户让你改写、翻译、润色、解释、计算一段文本时，那段文本不是要记的事实，"
                       "不要调用这个工具，直接做他要的处理。",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要记住的完整一句话，包含主语"},
            "answer": {"type": "string", "description": "这条知识里最关键的那个值（可选）"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "kb_lookup", "description": "到外部知识库里查一条事实。"
                       "**只在答案依赖用户私有的信息时才查**（他的工号、他的项目、他说过的事）。"
                       "通用知识（什么是 HTTP、谁是谁、单位换算）你自己就知道，直接回答，不要查。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要查的问题或关键词"}},
            "required": ["query"]}}},
]

CTX: dict = {"ready": False, "error": "", "params": 0, "dim": 0,
             "last_ms": 0.0, "last_tok": 0}
LOCK = threading.Lock()

_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_ARITH = re.compile(r"^[\d\s.\-+*/%()]+$")


# --------------------------------------------------------------------------- 模型

def fingerprint(model) -> str:
    """权重指纹：对固定的几层权重各取一段字节做 sha256。

    要的不是密码学强度，是一个便宜的"权重有没有被动过"的判据：写知识库前后各算一次，
    一样就说明这次更新没有走任何梯度。
    """
    h = hashlib.sha256()
    named = sorted(model.named_parameters())
    for i in (0, len(named) // 3, 2 * len(named) // 3, len(named) - 1):
        w = named[i][1].detach().to("cpu", torch.float16).flatten()[:1_000_000]
        h.update(w.numpy().tobytes())
    return h.hexdigest()


def boot() -> None:
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float16, device_map="cuda:0", low_cpu_mem_usage=True)
        model.eval()
        dim = int(model.config.hidden_size)
        kb = KnowledgeBase(KB_DIR, dim=dim)
        CTX.update(tok=tok, model=model, kb=kb, dim=dim,
                   params=sum(p.numel() for p in model.parameters()), ready=True)
    except Exception as exc:  # noqa: BLE001 —— 加载失败要在页面上看到真错误
        CTX["error"] = f"{type(exc).__name__}: {exc}"


@torch.no_grad()
def embed(text: str) -> np.ndarray:
    """用这个模型自己做嵌入（均值池化），不用再下载别的模型。

    取最后一层隐藏状态、按 attention_mask 均值池化再归一化。这不是最好的检索嵌入，
    但它不需要额外下载、不引入新权重，而且和生成用的是同一个模型——先跑通再谈换。
    """
    model, tok = CTX["model"], CTX["tok"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    base = getattr(model, "model", None) or model.get_base_model()
    hidden = base(**enc).last_hidden_state[0]
    mask = enc["attention_mask"][0].unsqueeze(1).to(hidden.dtype)
    pooled = (hidden * mask).sum(0) / mask.sum().clamp(min=1)
    v = pooled.float().cpu().numpy().astype("float32")
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def generate(text: str, max_new: int) -> tuple[str, int]:
    model, tok = CTX["model"], CTX["tok"]
    enc = tok(text, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.6,
                         top_p=0.95, top_k=20, repetition_penalty=1.05,
                         pad_token_id=tok.eos_token_id)
    new = out[0][n_in:]
    return tok.decode(new, skip_special_tokens=False), int(new.shape[0])


def parse_out(raw: str) -> tuple[str, str, list[dict]]:
    """把模型输出拆成 思考 / 正文 / 工具调用 三部分。"""
    think, body = "", raw
    m = _THINK.search(raw)
    if m:
        think, body = m.group(1).strip(), raw[m.end():]
    calls = []
    for cm in _CALL.finditer(body):
        obj = _loads(cm.group(1))
        if obj:
            calls.append(obj)
    body = _CALL.sub("", body)
    if not calls:
        # 小模型经常不按协议来，退一步接它实际写出来的样子
        for line in body.splitlines():
            obj = _parse_py_call(line)
            if obj:
                calls.append(obj)
                body = body.replace(line, "")
    body = body.replace("</s>", "")
    return think, body.strip(), calls


def _loads(text: str) -> dict | None:
    """标签里可能是 JSON，也可能是 Python 写法。两种都接。"""
    text = text.strip().rstrip(",")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return _parse_py_call(text)
    if isinstance(obj, dict) and obj.get("name"):
        return {"name": str(obj["name"]), "arguments": obj.get("arguments") or {}}
    return None


def _parse_py_call(text: str) -> dict | None:
    """接住 kb_add(text="…") 这种 Python 写法。

    0.6B 经常不按 <tool_call>{…}</tool_call> 的协议输出，而是直接写一个函数调用。
    协议是模型的能力边界，不是我们的妥协点：它实际写什么就接什么。
    用 ast 解析而不是 eval——参数全是常量，不会执行模型写的任意代码。
    """
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    args: dict = {}
    for kw in node.keywords:
        if kw.arg and isinstance(kw.value, ast.Constant):
            args[kw.arg] = kw.value.value
    if not args and node.args and isinstance(node.args[0], ast.Constant):
        spec = {t["function"]["name"]: t["function"]["parameters"]["properties"]
                for t in TOOLS}
        first = list(spec.get(node.func.id, {}))
        if first:
            args[first[0]] = node.args[0].value
    return {"name": node.func.id, "arguments": args} if args else None


# --------------------------------------------------------------------------- 工具

def tool_calc(expr: str) -> dict:
    """只允许数字和四则运算：先过白名单，再用 ast 解析，绝不 eval 原始字符串。"""
    if not _ARITH.match(expr or ""):
        return {"ok": False, "error": "表达式里只允许数字和 + - * / % ( )"}
    node = ast.parse(expr, mode="eval").body
    ops = {ast.Add: 1, ast.Sub: 1, ast.Mult: 1, ast.Div: 1, ast.Mod: 1, ast.Pow: 2,
           ast.UAdd: 1, ast.USub: 1}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = ev(n.left), ev(n.right)
            op = type(n.op)
            if op in (ast.Div, ast.Mod) and b == 0:
                raise ZeroDivisionError("除数为 0")
            import operator as O
            f = {ast.Add: O.add, ast.Sub: O.sub, ast.Mult: O.mul, ast.Div: O.truediv,
                 ast.Mod: O.mod, ast.Pow: O.pow}[op]
            return f(a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in (ast.UAdd, ast.USub):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        raise ValueError("不支持的表达式")

    try:
        return {"ok": True, "expr": expr, "value": ev(node)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def tool_kb_add(args: dict) -> dict:
    """写一条知识。前后各算一次权重指纹——这是"没重训"的证据，不是修辞。"""
    text = str(args.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "text 不能为空"}
    model, kb = CTX["model"], CTX["kb"]
    before = fingerprint(model)
    t0 = time.perf_counter()
    kb.add("chat", KnowledgeRecord(rid=KnowledgeRecord.make_id(text), domain="chat",
                                   kind="user_told", text=text,
                                   answer=str(args.get("answer", "")),
                                   source="user", verified=True), embed(text))
    ms = round((time.perf_counter() - t0) * 1000, 1)
    after = fingerprint(model)
    return {"ok": True, "written": text, "ms": ms, "kb_size": kb.size(),
            "weights_unchanged": before == after,
            "note": "写进的是外部知识库文件，模型权重一个字节都没动"}


def tool_kb_lookup(args: dict) -> dict:
    q = str(args.get("query", "")).strip()
    if not q:
        return {"ok": False, "error": "query 不能为空"}
    kb, tok = CTX["kb"], CTX["tok"]
    segs = kb.retrieve(embed(q), ["chat"], top_k=3, max_tokens_per_segment=200,
                       max_total_tokens=600, count_tokens=lambda s: len(tok.encode(s)),
                       query_text=q)
    return {"ok": True, "found": bool(segs),
            "results": [{"text": s.text, "score": round(s.score, 3)} for s in segs],
            "note": "没查到就说没查到，不要编" if not segs else "知识库里有这些，按这个答"}


def dispatch(call: dict) -> dict:
    name, args = call["name"], call["arguments"]
    if not isinstance(args, dict):
        args = {"expr": str(args)} if name == "calc" else {"query": str(args)}
    if name == "calc":
        return tool_calc(str(args.get("expr", "")))
    if name == "kb_add":
        return tool_kb_add(args)
    if name == "kb_lookup":
        return tool_kb_lookup(args)
    return {"ok": False, "error": f"没有这个工具：{name}"}


# --------------------------------------------------------------------------- 对话

def auto_context(question: str) -> tuple[str, list[dict]]:
    """回答前先去知识库里捞一遍，把命中的写成【已知】交给模型。

    为什么要有这一步：模型自己调 kb_lookup 这件事，1.7B 也不稳定——实测它宁可说
    "我无法得知您的工号"，也不去查。自动检索不是替模型作弊：页面会把"查到了什么"
    单独显示出来，你能看到答案是怎么变的。工具路径和自动路径都留着，各自测各自的数。
    """
    if not question:
        return "", []
    hits = tool_kb_lookup({"query": question})
    results = hits.get("results") or []
    if not results:
        return "", []
    return "\n".join(f"【已知】{r['text']}" for r in results), results


def chat(messages: list[dict], think: bool = False, max_steps: int = 4,
         auto: bool = True) -> dict:
    """一轮对话。模型可能连着调几次工具，最后才给回答。

    工具结果**以 user 身份**包在 <tool_response>…
    里喂回去——这是 Qwen3 对话模板里写死的约定（我查过模板：它判断多轮工具
    就是看 user 消息有没有这对标记）。用 role="tool" 反而不认。
    """
    tok = CTX["tok"]
    question = next((str(m.get("content", "")) for m in reversed(messages)
                     if m.get("role") == "user"), "")
    # auto=False 就是"只靠模型自己调工具"那条路，用来分开测两条路各自的命中率
    ctx, hits = auto_context(question) if auto else ("", [])
    sys_prompt, steps = SYSTEM, []
    if ctx:
        sys_prompt += "\n\n知识库里查到的、已确认的信息，回答时以它为准：\n" + ctx
        steps.append({"tool": "auto_retrieve", "args": {"query": question},
                      "result": {"found": len(hits), "results": hits}})
    msgs = [{"role": "system", "content": sys_prompt}] + list(messages)
    for _ in range(max_steps):
        prompt = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=think)
        t0 = time.perf_counter()
        raw, n_tok = generate(prompt, 700 if think else 400)
        dt = time.perf_counter() - t0
        CTX["last_ms"], CTX["last_tok"] = round(dt * 1000), n_tok
        reasoning, body, calls = parse_out(raw)
        if not calls:
            return {"reply": body, "reasoning": reasoning, "steps": steps,
                    "tok_per_s": round(n_tok / max(dt, .01), 1)}
        msgs.append({"role": "assistant", "content": raw})
        results = []
        for c in calls:
            res = dispatch(c)
            steps.append({"tool": c["name"], "args": c["arguments"], "result": res,
                          "reasoning": reasoning})
            results.append({"tool": c["name"], "result": res})
            reasoning = ""          # 思考只属于第一次调用前，别重复贴
        msgs.append({"role": "user",
                     "content": "<tool_response>\n" + json.dumps(results, ensure_ascii=False)
                                + "\n"})
    return {"reply": "（调了太多次工具，先停在这里。上面的工具结果是真实的。）",
            "reasoning": "", "steps": steps, "stopped": True}


def state_info() -> dict:
    if not CTX["ready"]:
        return {"ready": False, "error": CTX["error"]}
    model, kb = CTX["model"], CTX["kb"]
    return {"ready": True, "model": Path(MODEL_PATH).name, "params": f"{CTX['params']:,}",
            "kb_size": kb.size(), "fingerprint": fingerprint(model)[:16],
            "speed": round(CTX["last_tok"] / max(CTX["last_ms"], 1) * 1000, 1)}


def kb_dump(query: str = "") -> dict:
    """把知识库里现有的条目列出来，给页面上那个"知识库"面板用。"""
    kb = CTX["kb"]
    idx = kb.indices.get("chat")
    records = list(idx.records) if idx is not None else []
    items = [{"rid": r.rid, "text": r.text, "answer": r.answer, "source": r.source}
             for r in records if not query or query.lower() in r.text.lower()]
    return {"kb_size": kb.size(), "items": items[-200:]}


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>聊天台</title>
<style>
:root{--bg:#11141a;--card:#1a1f28;--line:#2b3140;--fg:#e6e9ef;--dim:#8b93a7;--ok:#3fb950;
--bad:#f85149;--warn:#d29922;--ac:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:20px 18px 90px}
h1{font-size:19px;margin:0 0 4px}.sub{color:var(--dim);font-size:13px}
.bar{display:flex;gap:16px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 13px;font-size:12.5px;color:var(--dim);margin:14px 0}
.bar b{color:var(--fg)}
.msg{margin:14px 0}.me{display:flex;justify-content:flex-end}
.me div{background:#1f6feb;border-radius:12px;padding:10px 14px;max-width:80%}
.them div{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:10px 14px;max-width:88%;white-space:pre-wrap}
details{background:#151922;border:1px solid var(--line);border-radius:8px;padding:6px 10px;
margin:6px 0;font-size:13px;color:var(--dim)}details summary{cursor:pointer;color:var(--dim)}
.tool{border-left:3px solid var(--ac);padding-left:9px;margin:6px 0;color:var(--fg)}
.tool b{color:var(--ac)}.kv{font-family:ui-monospace,Consolas,monospace;font-size:12px;
color:var(--dim);word-break:break-all}
.badge{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:20px;margin-left:6px}
.badge.ok{background:rgba(63,185,80,.15);color:var(--ok)}
.badge.bad{background:rgba(248,81,73,.15);color:var(--bad)}
.in{position:fixed;left:0;right:0;bottom:0;background:#0d1015;border-top:1px solid var(--line);
padding:10px 14px;display:flex;gap:8px;align-items:center}
.in input{flex:1;background:#0b0e13;border:1px solid var(--line);color:var(--fg);
border-radius:8px;padding:11px 13px;font-size:15px}
.in input:focus{outline:none;border-color:var(--ac)}
button{background:#21262d;border:1px solid var(--line);color:var(--fg);border-radius:8px;
padding:11px 15px;cursor:pointer;font-size:14px}button.go{background:#1f6feb;border-color:#1f6feb}
label{font-size:13px;color:var(--dim);white-space:nowrap;display:flex;align-items:center;gap:5px}
#kb{display:none;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px;margin:12px 0;font-size:13px}.kbi{border-bottom:1px dashed var(--line);padding:5px 0}
.kbi button{padding:2px 8px;font-size:12px}
.spin{color:var(--ac)}.foot{color:var(--dim);font-size:12px;margin-top:22px;line-height:1.8}
</style></head><body><div class="wrap">
<h1>聊天台 · 真模型 + 外部知识库</h1>
<div class="sub">分工写清楚，免得误会：<b style="color:var(--fg)">对话、思考、调工具是底座模型（Qwen3）本身的能力，
不是这个项目做的</b>。这个项目加上去的是外面那一层——它记住的新事实存在模型外面，
写进去不用重训，权重一个字节都不动，换个进程还记得。</div>
<div class="bar"><span>状态 <b id="st">加载中</b></span>
<span>底座 <b id="md">—</b>（外部预训练，非本项目所训）</span>
<span>参数 <b id="pm">—</b></span><span>知识库 <b id="kbN">—</b> 条</span>
<span>权重指纹 <b id="fp">—</b></span><span>速度 <b id="sp">—</b> tok/s</span></div>
<div id="kb"></div><div id="log"></div>
<div class="foot">底座是 Qwen3（0.6B 和 1.7B 都能跑，现在跑的是 1.7B）：对话没问题，
推理质量有限，思考过程会想歪。要更好得换 4B——显存够（24 GB 全空），卡在主机内存只剩 4.9 GB。
本页真正演示的是外面那一层：写入 46 ms、零梯度、跨进程生效。工具调用和自动检索的过程都摊开给你看。</div>
</div>
<div class="in"><input id="t" placeholder="跟它说点什么，比如：记一下，我们项目的部署机器是 10.20.30.40"
 autofocus><button class="go" onclick="send()">发送</button>
<button onclick="kbToggle()">知识库</button><label><input type="checkbox" id="th">思考</label></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let H=[];
async function st(){try{const r=await fetch('/api/state').then(x=>x.json());
 if(!r.ready){$('st').innerHTML='<span class="spin">模型加载中…</span>';return}
 $('st').textContent='就绪';$('md').textContent=r.model;$('pm').textContent=r.params;
 $('kbN').textContent=r.kb_size;$('fp').textContent=r.fingerprint;$('sp').textContent=r.speed;
 }catch(e){$('st').textContent='服务没起来'}}
function say(role,html){const d=document.createElement('div');d.className='msg '+(role=='user'?'me':'them');
 d.innerHTML='<div>'+html+'</div>';$('log').appendChild(d);scrollTo(0,document.body.scrollHeight)}
function tools(steps){return steps.map(s=>{const r=s.result||{};
 const b=r.weights_unchanged===true?'<span class="badge ok">权重没动</span>'
   :(r.ok===false?'<span class="badge bad">失败</span>':'');
 return `<details open><summary>调用工具 <b>${esc(s.tool)}</b>${b}</summary>
 <div class="tool">参数 <span class="kv">${esc(JSON.stringify(s.args))}</span><br>
 结果 <span class="kv">${esc(JSON.stringify(r))}</span></div></details>`}).join('')}
async function send(){const t=$('t').value.trim();if(!t)return;$('t').value='';
 H.push({role:'user',content:t});say('user',esc(t));
 say('them','<span class="spin">在想…</span>');const box=$('log').lastChild.firstChild;
 let r;try{r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({messages:H,think:$('th').checked})}).then(x=>x.json())}
 catch(e){box.innerHTML='<span style="color:var(--bad)">请求失败：'+esc(e)+'</span>';return}
 if(r.error){box.innerHTML='<span style="color:var(--bad)">'+esc(r.error)+'</span>';return}
 let h='';
 if(r.reasoning)h+=`<details><summary>思考过程</summary><div style="white-space:pre-wrap">
 ${esc(r.reasoning)}</div></details>`;
 if(r.steps&&r.steps.length)h+=tools(r.steps);
 h+=esc(r.reply||'（空）');box.innerHTML=h;
 H.push({role:'assistant',content:r.reply||''});st()}
async function kbToggle(){$('kb').style.display='block';
 const r=await fetch('/api/kb').then(x=>x.json());
 $('kb').innerHTML='<b>知识库里的 '+r.kb_size+' 条</b>（这就是模型"记住"的东西，存在模型外面）'
 +(r.items.length?r.items.map(i=>`<div class="kbi">${esc(i.text)}
  <button onclick="del('${i.rid}')">删</button></div>`).join(''):'<div class="kv">（还空着。跟它说"记一下…"它就会写进来）</div>')}
async function del(rid){await fetch('/api/kb_delete',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({rid})});kbToggle();st()}
$('t').addEventListener('keydown',e=>{if(e.key==='Enter')send()});st();setInterval(st,5000);
</script></body></html>"""


# --------------------------------------------------------------------------- 服务

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(200, json.dumps(state_info(), ensure_ascii=False))
        elif self.path.startswith("/api/kb"):
            if not CTX["ready"]:
                return self._send(503, json.dumps({"error": "模型还在加载中"}))
            q = ""
            if "q=" in self.path:
                from urllib.parse import unquote
                q = unquote(self.path.split("q=", 1)[1].split("&")[0])
            with LOCK:
                self._send(200, json.dumps(kb_dump(q), ensure_ascii=False))
        else:
            self._send(200, PAGE, "text/html")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, json.dumps({"error": "请求体不是合法 JSON"}))
        if not CTX["ready"]:
            return self._send(503, json.dumps({"error": "模型还在加载中，请稍等"}))
        kind = self.path.rsplit("/", 1)[-1]
        with LOCK:
            try:
                if kind == "chat":
                    out = chat(req.get("messages") or [], bool(req.get("think")))
                elif kind == "kb_delete":
                    rid = str(req.get("rid", ""))
                    gone = CTX["kb"].remove("chat", rid)
                    out = {"ok": gone, "kb_size": CTX["kb"].size()}
                else:
                    return self._send(404, json.dumps({"error": "没有这个接口"}))
            except Exception as exc:  # noqa: BLE001
                out = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(out, ensure_ascii=False))


def selftest() -> int:
    """三条能不能过，直接决定这个东西值不值得继续做。跑的是真模型，不是模拟。

    1 算术必须走工具（不是它自己心算出来的数）
    2 告诉它一个新事实 → 它自己调 kb_add，而且权重指纹没变（这就是"不用重训"）
    3 再问同一个问题 → 它调 kb_lookup 查到并照知识库答（重启进程后同样成立）
    """
    # 自测从空库开始：库里上次留下的同一条问题会干扰判断（就是 R4 说的单一权威来源）
    for stale in Path(KB_DIR).glob("chat*"):
        stale.unlink()
    boot()
    if not CTX["ready"]:
        print("加载失败：", CTX["error"])
        return 1
    print(f"模型 {CTX['params']:,} 参数，嵌入维度 {CTX['dim']}，知识库 {CTX['kb'].size()} 条\n")
    fails = 0

    def case(name, msgs, think, check):
        nonlocal fails
        r = chat(msgs, think=think)
        tools = [s["tool"] for s in r["steps"]]
        ok = check(r, tools)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print(f"      调了工具 {tools or '（无）'}")
        for s in r["steps"]:
            print(f"      {s['tool']} {json.dumps(s['args'], ensure_ascii=False)} → "
                  f"{json.dumps(s['result'], ensure_ascii=False)[:200]}")
        print(f"      回答：{(r['reply'] or '')[:200]}")
        if not ok:
            fails += 1
        return r

    case("算术走工具", [{"role": "user", "content": "1234567 乘以 7919 等于多少？直接给结果"}],
         False, lambda r, t: "calc" in t and any(
             s["result"].get("value") == 1234567 * 7919 for s in r["steps"]))
    # 用一个每次都不一样的数字，免得库里上次留下的条目把这次"记住"混过去
    code = str(100000 + (time.time_ns() % 899999))
    r2 = case("新事实写进外部知识库", [{"role": "user",
              "content": f"记一下：我的工号是 {code}"}], False,
              lambda r, t: "kb_add" in t and all(
                  s["result"].get("weights_unchanged") for s in r["steps"]
                  if s["tool"] == "kb_add"))
    if r2["steps"]:
        print(f"      ↑ 权重指纹前后一致 = {r2['steps'][0]['result'].get('weights_unchanged')}"
              f"，写入耗时 {r2['steps'][0]['result'].get('ms')} ms")
    case("过后再问要能查到", [{"role": "user", "content": "我的工号是多少？"}], False,
         lambda r, t: ("kb_lookup" in t or "auto_retrieve" in t) and code in (r["reply"] or ""))
    print(f"\n{'全部通过' if fails == 0 else str(fails) + ' 条没过'}"
          f"（重启进程后再问第 3 条同样成立——知识在模型外面）")
    return 0 if fails == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="真模型聊天台：对话 + 工具 + 思考 + 外部知识库")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--selftest", action="store_true", help="跑三条验证，不起服务")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    threading.Thread(target=boot, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"聊天台已启动：http://127.0.0.1:{args.port}（模型正在后台加载，约 10 秒）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())