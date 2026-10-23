"""KSR 演示台的网页前端：只用 Python 标准库，不装任何包。

为什么要有这个：命令行演示台要终端，而这个环境是通过浏览器进来的，终端看不见。
这个页面把同一套逻辑搬到浏览器里，而且刻意把关键的两件事**并排**显示——
"不检索时模型答什么"和"带检索时模型答什么"——因为只看最终答案，就把"检索起了作用"
和"模型本来就会"混成一回事了。

用法：
    python tools/webui.py --work work-conflict --port 8765
    然后浏览器打开 http://127.0.0.1:8765

模型只加载一次（约 40 秒），加载期间页面会显示进度。请求是串行的（一把锁），
因为这是个 220 万参数的原型，两个请求并发跑只会互相拖慢。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if "--work" in sys.argv:
    os.environ["KSR_WORK"] = sys.argv[sys.argv.index("--work") + 1]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from play import load, probe, write_fact  # noqa: E402
import ksr.experiment as E  # noqa: E402

CTX = {"model": None, "tok": None, "embed": None, "kb": None, "corpus": None, "cfg": None,
       "params": 0, "ready": False, "error": ""}
LOCK = threading.Lock()

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KSR 演示台</title>
<style>
 :root{--bg:#12141a;--card:#1b1f28;--line:#2b3140;--fg:#e6e9ef;--dim:#8b93a7;
       --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
 .wrap{max-width:820px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
 .bar{display:flex;gap:18px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
      border-radius:10px;padding:12px 14px;font-size:13px;color:var(--dim);margin-bottom:20px}
 .bar b{color:var(--fg);font-weight:600}
 .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
 input{background:#0e1015;border:1px solid var(--line);color:var(--fg);border-radius:8px;
       padding:10px 12px;font-size:15px;min-width:0}
 input:focus{outline:none;border-color:var(--accent)}
 #q{flex:1;min-width:220px}
 button{background:#21262d;border:1px solid var(--line);color:var(--fg);border-radius:8px;
        padding:10px 16px;font-size:14px;cursor:pointer}
 button:hover{border-color:var(--accent)}
 button.go{background:#1f6feb;border-color:#1f6feb}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;
       margin-top:14px}
 .card h3{margin:0 0 10px;font-size:13px;color:var(--dim);font-weight:600;
          text-transform:uppercase;letter-spacing:.06em}
 .seg{font-family:ui-monospace,Consolas,monospace;font-size:13px;color:var(--fg);
      padding:6px 0;border-bottom:1px dashed var(--line)}
 .seg:last-child{border:0}
 .score{color:var(--dim);float:right}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 .ans{border:1px solid var(--line);border-radius:8px;padding:12px}
 .ans .v{font-size:26px;font-weight:600;font-family:ui-monospace,Consolas,monospace}
 .ans .l{font-size:12px;color:var(--dim);margin-bottom:6px}
 .ans.win{border-color:var(--ok)} .ans.win .v{color:var(--ok)}
 .tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:20px;margin-top:8px}
 .tag.ok{background:rgba(63,185,80,.15);color:var(--ok)}
 .tag.bad{background:rgba(248,81,73,.15);color:var(--bad)}
 .kv{font-size:13px;color:var(--dim);margin:4px 0}
 .kv b{color:var(--fg)}
 .note{border-left:3px solid var(--warn);background:rgba(210,153,34,.08);padding:10px 12px;
       border-radius:0 8px 8px 0;font-size:13px;margin-top:12px}
 .foot{color:var(--dim);font-size:12px;margin-top:26px;line-height:1.8}
 code{background:#0e1015;padding:1px 5px;border-radius:4px;font-size:12px}
 .spin{color:var(--accent)}
</style></head><body><div class="wrap">
<h1>KSR 演示台</h1>
<div class="sub">往知识库里加一条事实，不重训、不产生梯度，模型立刻能答对。
 这个页面同时把失败也显示出来——它不演示路由选专家（实测把全部专家清零，答案一题都不变）。</div>

<div class="bar">
  <span>状态 <b id="st">加载中…</b></span>
  <span>模型 <b id="pm">—</b></span>
  <span>知识库 <b id="kb">—</b> 条</span>
  <span>权重指纹 <b id="fp">—</b></span>
</div>

<div class="row">
  <input id="q" placeholder="输入端口号，或直接把问题粘进来（例如 9005）" autofocus>
  <input id="svc" placeholder="服务名（加/改 时填，例如 svc07）" style="flex:1;min-width:180px">
</div>
<div class="row">
  <button class="go" onclick="call('ask')">问</button>
  <button onclick="call('write',false)">加一条</button>
  <button onclick="call('write',true)">改</button>
  <button onclick="call('del')">删掉</button>
  <span class="sub" id="hint">提示：加 9100 svc00 → 立刻再问，就会答对了</span>
</div>

<div id="out"></div>

<div class="foot">
这是 <b>220 万参数的原型</b>，任务是合成的「端口 → 服务」，不是聊天模型。<br>
「混合分」是稠密相似度<b>加上符号（编号）精确命中的加成</b>，所以会超过 1；
排序先按符号命中数、再按这个分数——这是库涨到几千条也不漏的原因。<br>
注入受三重硬预算约束：最多 2 段、单段 48 token、总共 96 token。
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function port(){const t=$('q').value.trim();const m=t.match(/\\d{3,7}/);return m?m[0]:t}
async function st(){
  const r=await fetch('/api/state').then(x=>x.json());
  if(r.error){$('st').textContent='加载失败：'+r.error;return}
  if(!r.ready){$('st').innerHTML='<span class="spin">模型加载中（约 40 秒）…</span>';return}
  $('st').textContent='就绪';$('pm').textContent=r.params;$('kb').textContent=r.kb_size;
  $('fp').textContent=r.fingerprint.slice(0,16)+'…';
}
async function call(kind,replace){
  if(!port()){$('out').innerHTML='<div class="card">填个端口号</div>';return}
  $('out').innerHTML='<div class="card"><span class="spin">算一下，几秒…</span></div>';
  const body=kind==='write'?{port:port(),svc:$('svc').value.trim(),replace:!!replace}:{port:port()};
  const r=await fetch('/api/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},
                                    body:JSON.stringify(body)}).then(x=>x.json());
  if(r.error){$('out').innerHTML='<div class="card">'+esc(r.error)+'</div>';return}
  render(r,kind);
}
function render(r,kind){
  const p=kind==='del'?r.probe:(r.probe||r);
  const w=r.write;
  let h='';
  if(r.removed){
    h+=`<div class="card"><h3>从知识库里删掉</h3>`
      +(r.removed.length?r.removed.map(t=>`<div class="kv">已删除：<b>${esc(t)}</b></div>`).join('')
        :'<div class="kv">没有找到这个端口的条目</div>')+'</div>';
  }
  if(w){
    h+=`<div class="card"><h3>写入知识库</h3>
      <div class="kv">写入 <b>${esc(w.text)}</b>　用时 <b>${w.ms} ms</b>　库现有 <b>${w.size}</b> 条</div>
      ${w.removed.map(t=>`<div class="kv">删掉旧条目：${esc(t)}</div>`).join('')}
      <div class="kv">权重指纹 <b>${w.fingerprint_before.slice(0,16)}… → ${w.fingerprint_after.slice(0,16)}…</b>
        ${w.weights_unchanged?'<span class="tag ok">没变，一个梯度都没走</span>':'<span class="tag bad">变了</span>'}</div>
      ${w.note?`<div class="note">${esc(w.note)}</div>`:''}</div>`;
  }
  if(!p)return;
  h+=`<div class="card"><h3>检索到的知识片段</h3>`;
  h+=p.segments.length
    ? p.segments.map(s=>`<div class="seg">${esc(s.text)}<span class="score">混合分 ${s.score}</span></div>`).join('')
      +`<div class="kv">注入 <b>${p.injected_tokens}</b> token　预算 <b>${p.budget}</b>　最多 2 段</div>`
      +(p.matched?'':`<div class="note">这些<b>都不是在讲端口 ${p.port}</b>——库里没有这条知识。`
        +'模型接下来要么照抄一条不相干的，要么猜。所以下面那个答案不能算"用了知识库"。</div>')
    : '<div class="kv">知识库里没有这条</div>';
  h+='</div>';
  const win=p.matched?(p.adopted?'right':(p.held_to_prior?'left':'none')):'none';
  const right=p.matched?('知识库里写着 '+esc(p.kb_answer||'—'))
    :'知识库里没有这个端口';
  const tag=p.matched
    ?(p.adopted?'<span class="tag ok">采纳了知识库</span>'
      :(p.held_to_prior?'<span class="tag bad">坚持自己的先验</span>':''))
    :'<span class="tag bad">没有这条知识</span>';
  h+=`<div class="card"><h3>模型答什么 —— 两边并排才看得出是谁在起作用</h3><div class="pair">
    <div class="ans ${win==='left'?'win':''}"><div class="l">不检索（模型自己的先验）</div>
      <div class="v">${esc(p.prior||'（空）')}</div></div>
    <div class="ans ${win==='right'?'win':''}"><div class="l">带检索（${right}）</div>
      <div class="v">${esc(p.answer||'（空）')}</div>${tag}</div>
  </div></div>`;
  $('out').innerHTML=h;
  st();
}
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')call('ask')});
st();setInterval(st,4000);
</script></body></html>"""


def state_info() -> dict:
    if not CTX["ready"]:
        return {"ready": False, "error": CTX["error"]}
    model, kb = CTX["model"], CTX["kb"]
    return {"ready": True, "params": f"{CTX['params']:,}", "kb_size": kb.size(),
            "fingerprint": E.model_fingerprint(model)}


def handle(kind: str, req: dict) -> dict:
    model, tok, embed, kb, corpus, cfg = (CTX["model"], CTX["tok"], CTX["embed"], CTX["kb"],
                                          CTX["corpus"], CTX["cfg"])
    port_raw = str(req.get("port", ""))
    m = re.search(r"\d{3,7}", port_raw)
    if not m:
        return {"error": "没看懂端口号。输入 3-7 位数字，或者把整句话粘进来。"}
    port = int(m.group())
    if kind == "ask":
        return probe(model, tok, embed, kb, cfg, port)
    if kind == "del":
        removed = []
        for dom, old in kb.find(f"端口 {port} 的服务是"):
            kb.remove(dom, old.rid)
            removed.append(old.text)
        return {"removed": removed,
                "probe": probe(model, tok, embed, kb, cfg, port)}
    # 写入
    svc = str(req.get("svc", "")).strip()
    if not svc:
        return {"error": "加/改 需要填服务名（例如 svc07）。"}
    note = ""
    if len(tok.encode(svc)) != 1:
        note = (f"「{svc}」不在模型词表里（会被拆成 {len(tok.encode(svc))} 个字符），"
                f"模型读不出来。知识库能补『哪个端口对应哪个服务』，"
                f"补不了『模型从没学过怎么读这个词』。")
    elif svc not in set(corpus["fact_table"].values()):
        note = (f"「{svc}」模型**认识**（是一个整 token），但这个槽位上它从没被要求抄过"
                f"这种形状的答案，接下来大概率不采纳——这不是读不出词，而是"
                f"**认账只在它见过的答案形状里成立**（实测过：加 9101 Panov，"
                f"检索命中了，模型照样不抄）。")
    out = {"write": write_fact(model, kb, embed, port, svc, bool(req.get("replace")))}
    out["write"]["note"] = note
    out["probe"] = probe(model, tok, embed, kb, cfg, port)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):        # 别把每次请求都打到控制台
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
        else:
            self._send(200, PAGE, "text/html")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, json.dumps({"error": "请求体不是合法 JSON"}))
        if not CTX["ready"]:
            return self._send(503, json.dumps({"error": "模型还在加载中，请等页面顶部变成「就绪」"}))
        kind = self.path.rsplit("/", 1)[-1]
        if kind not in ("ask", "write", "del"):
            return self._send(404, json.dumps({"error": "没有这个接口"}))
        with LOCK:                    # 一次只跑一个请求：并发只会互相拖慢
            try:
                out = handle(kind, req)
            except Exception as exc:  # noqa: BLE001 —— 页面上要看到真错误，不是转圈
                out = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(out, ensure_ascii=False))


def boot() -> None:
    try:
        model, tok, embed, kb, corpus, cfg = load()
        CTX.update(model=model, tok=tok, embed=embed, kb=kb, corpus=corpus, cfg=cfg,
                   params=sum(p.numel() for p in model.parameters()), ready=True)
    except Exception as exc:  # noqa: BLE001
        CTX["error"] = f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="KSR 演示台的网页前端（只用标准库）")
    ap.add_argument("--work", default="work-conflict")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("KSR_THREADS", "2")))
    threading.Thread(target=boot, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"演示台已启动：http://127.0.0.1:{args.port}  （模型正在后台加载，约 40 秒）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
