r"""造 SFT 数据：教模型「什么时候该调哪个工具、什么时候别调」。

为什么要自己造数据：测出来的两个短板都是**判断**问题，不是知识问题——
该查的时候不查（自己去查 4/6）、不该写的时候把要改写的句子写进库（误触发 1/3）。
这两种都没训过，光靠改工具描述只能修一半。

格式必须和推理时完全一致，否则训了白训。所以这里直接复用 chatui 的 SYSTEM 和 TOOLS，
工具调用写成 `<tool_call>{json}</tool_call>`，工具结果**以 user 身份**包在
`<tool_response>…</tool_response>` 里——这三条都是 Qwen3 模板里写死的约定（查过模板）。

用法：
    python tools/sftdata.py --n 40000 --out work-chat/sft/train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import chatui as C  # noqa: E402  只为拿 SYSTEM / TOOLS，不加载模型

# 评测集（tools/chatbench.py）里用过的实体，训练里一律避开，不然分数是背出来的
HOLDOUT = {"483920", "10.14.7.22", "3 月 12", "secret/db/prod", "灯塔", "王工", "李工",
           "客服系统", "计费系统", "Kx7-2213", "8 月 3", "HTTP"}

SURNAMES = ["陈工", "刘工", "赵工", "孙工", "周工", "吴工", "郑工", "冯工", "蒋工", "韩工"]
SYSTEMS = ["订单系统", "风控系统", "监控平台", "网关", "数据仓库", "搜索服务",
           "对账系统", "权限中心", "推送服务", "报表平台"]
CODENAMES = ["北斗", "青鸟", "长庚", "天枢", "望舒", "南斗", "玄戈", "建木", "白泽", "承影"]
PATHS = ["vault/secret/pay/prod", "s3://cfg-bucket/db/prod", "/etc/app/secrets/db",
         "consul://kv/prod/db", "kms://rds-credentials", "gitops/secrets/db/prod"]
MACHINES = ["10.31.4.18", "172.19.8.55", "192.168.44.7", "10.8.90.2", "172.20.13.61"]
DATES = ["4 月 7 日", "9 月 23 日", "11 月 2 日", "1 月 19 日", "6 月 30 日", "2 月 14 日"]
CODES = ["Gw4-8821", "Tq9-1174", "Hn2-6650", "Zx8-3092", "Bd5-7714", "Lm1-4408"]
TIMES = ["每周二上午十点", "每月最后一个工作日", "每天下午六点", "隔周周四下午三点"]
PLACES = ["三楼打印区", "B2 层储物柜 17 号", "前台左侧抽屉", "园区南门门卫室", "机房 A 列机柜顶"]

# 通用概念：用来造「不该调工具」的常识问答（这些模型本来就知道）
DEFINITIONS = {
    "什么是 HTTP": "HTTP 是应用层协议，用于在客户端和服务器之间传输超文本。",
    "什么是 TCP": "TCP 是面向连接的传输层协议，提供可靠、按序的字节流传输。",
    "什么是 DNS": "DNS 是把域名解析为 IP 地址的分布式命名系统。",
    "什么是 HTTPS": "HTTPS 是在 HTTP 之上加了 TLS 加密，提供机密性和完整性。",
    "什么是 REST": "REST 是一种架构风格，用无状态的、面向资源的接口来设计服务。",
    "什么是死锁": "死锁是多个事务互相持有对方需要的资源、又都在等对方释放，谁都无法推进。",
    "什么是索引": "索引是数据库为某些列建立的额外结构，用空间换查询速度。",
    "什么是幂等": "幂等指同一操作执行一次和执行多次，产生的效果相同。",
    "什么是容器": "容器是操作系统级的隔离机制，把进程及其依赖打包成可重复运行的单元。",
    "什么是消息队列": "消息队列在生产者与消费者之间传递消息，用来解耦和削峰。",
    "什么是缓存穿透": "缓存穿透是查询一个根本不存在的数据，缓存不命中，请求反复落到数据库上。",
    "什么是蓝绿发布": "蓝绿发布是准备两套环境，流量从一套整体切到另一套，回滚就是切回去。",
    "什么是分页查询": "分页查询是把结果按页取，避免一次把全部数据读进内存。",
    "什么是事务隔离级别": "事务隔离级别规定一个事务能看到其他事务什么状态的数据，从低到高依次是读未提交、读已提交、可重复读、串行化。",
    "什么是 JWT": "JWT 是一种自包含的令牌格式，由头部、载荷和签名三段组成，服务端不必保存会话。",
    "什么是内存泄漏": "内存泄漏是程序不再使用的内存没有被释放，可用内存随时间持续减少。",
}
# 单位换算：同样是「本来就知道」的，用来教它别乱调工具
CONVERTS = [("1 千米等于多少米", "1000 米"), ("1 小时等于多少秒", "3600 秒"),
            ("1 公斤等于多少克", "1000 克"), ("1 分钟等于多少秒", "60 秒"),
            ("1 吨等于多少千克", "1000 千克"), ("1 天等于多少小时", "24 小时"),
            ("1 吉字节等于多少兆字节", "1024 兆字节"), ("1 周等于多少天", "7 天"),
            ("1 安培等于多少毫安", "1000 毫安"), ("1 伏等于多少毫伏", "1000 毫伏"),
            ("1 元等于多少分", "100 分"), ("1 米等于多少厘米", "100 厘米")]


# --------------------------------------------------------------------- 消息拼装

def _call(name: str, args: dict) -> str:
    return ("<tool_call>\n" + json.dumps({"name": name, "arguments": args},
                                     ensure_ascii=False) + "\n</tool_call>")


def _obs(result: dict) -> str:
    return "<tool_response>\n" + json.dumps(result, ensure_ascii=False) + "\n"


def _msgs(user: str, turns: list) -> list:
    out = [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": user}]
    for text, obs in turns:
        out.append({"role": "assistant", "content": text})
        if obs:
            out.append({"role": "user", "content": obs})
    return out


# --------------------------------------------------------------------- 各类样本

def gen_recall(rng):
    """库里有一条事实 → 应该调 kb_lookup → 照知识库答。这是修「自己去查 4/6」的主力。"""
    which = rng.randrange(9)
    if which == 0:
        v = f"{rng.randint(100000, 999999)}"
        fact, q, a = f"我的工号是 {v}", "我的工号是多少？", f"你的工号是 {v}。"
    elif which == 1:
        v = rng.choice(MACHINES)
        fact, q, a = f"我们项目的部署机器是 {v}", "我们项目的部署机器是哪台？", f"部署机器是 {v}。"
    elif which == 2:
        s, c = rng.choice(SYSTEMS), rng.choice(CODENAMES)
        fact, q, a = f"我们内部把{s}叫作{c}", f"{s}内部叫什么代号？", f"{s}内部叫作{c}。"
    elif which == 3:
        s, p = rng.choice(SYSTEMS), rng.choice(PATHS)
        fact, q, a = f"{s}的数据库连接串存放在 {p}", f"{s}的数据库连接串存放在哪里？", f"存放在 {p}。"
    elif which == 4:
        s, w = rng.choice(SYSTEMS), rng.choice(SURNAMES)
        fact, q, a = f"{w}负责{s}", f"{s}是谁在负责？", f"由{w}负责。"
    elif which == 5:
        d = rng.choice(DATES)
        fact, q, a = f"我爱人的生日是 {d}", "我爱人的生日是哪天？", f"是 {d}。"
    elif which == 6:
        v = rng.choice(CODES)
        fact, q, a = f"我们的预发环境密码是 {v}", "预发环境的密码是什么？", f"是 {v}。"
    elif which == 7:
        t = rng.choice(TIMES)
        fact, q, a = f"我们的约定是周报{t}提交", "周报什么时候交？", f"{t}提交。"
    else:
        p = rng.choice(PLACES)
        fact, q, a = f"备用门禁卡放在{p}", "备用门禁卡放在哪里？", f"在{p}。"
    obs = {"tool": "kb_lookup", "result": {"ok": True, "found": True, "results": [
        {"text": fact, "score": round(rng.uniform(0.88, 0.99), 3)}], "note": "知识库里有这些，按这个答"}}
    return _msgs(q, [(_call("kb_lookup", {"query": q}), _obs(obs)), (a, None)]), "recall"


def gen_refuse(rng):
    """库里没有 → 调了 kb_lookup → 明确说没查到。教它「没证据时要会拒答」。

    问题里必须带具体主题：不加主题整类只有三五句话，去重之后只剩个位数
    （实测过：3000 条里 refuse 只出来 23 条）。
    """
    s = rng.choice(SYSTEMS)
    topic, asks = rng.choice([
        (s, [f"{s}是谁在负责？", f"{s}归谁管？", f"谁负责{s}？"]),
        (s, [f"{s}的内部代号是什么？", f"{s}内部叫什么？", f"{s}的代号是啥？"]),
        (s, [f"{s}的值班表在哪里？", f"{s}的值班表放哪了？"]),
        (s, [f"{s}的告警阈值是多少？", f"{s}的告警阈值设的多少？"]),
        (s, [f"{s}的数据库连接串存放在哪里？", f"{s}的连接串放在哪？"]),
        ("工号", ["我的工号是多少？", "我工号几号来着？", "还记得我的工号吗？"]),
        ("备用门禁卡", ["备用门禁卡放在哪里？", "备用门禁卡放哪了？"]),
        ("我爱人的生日", ["我爱人的生日是哪天？", "我说过我爱人生日是什么时候？"]),
        ("部署机器", ["我上次说的那台部署机器是哪台？", "我们的部署机器是哪台？"]),
        ("预发环境密码", ["预发环境的密码是什么？", "预发环境密码多少？"])])
    q = rng.choice(asks)
    obs = {"tool": "kb_lookup", "result": {"ok": True, "found": False, "results": [],
                                           "note": "没查到就说没查到，不要编"}}
    a = rng.choice([
        "知识库里没有这条，我没查到。",
        f"我没查到——知识库里没有关于{topic}的记录。你要是告诉我，我可以记下来。",
        "没有查到相关记录。需要的话我可以把它记进知识库，以后就能查到了。"])
    return _msgs(q, [(_call("kb_lookup", {"query": q}), _obs(obs)), (a, None)]), "refuse"


def gen_write(rng):
    """用户在陈述一个新事实 → 应该调 kb_add，然后只回「记下了」。"""
    which = rng.randrange(5)
    if which == 0:
        fact = f"我的工号是 {rng.randint(100000, 999999)}"
    elif which == 1:
        fact = f"我们项目的部署机器是 {rng.choice(MACHINES)}"
    elif which == 2:
        fact = f"我们内部把{rng.choice(SYSTEMS)}叫作{rng.choice(CODENAMES)}"
    elif which == 3:
        fact = f"我家的猫生日是 {rng.choice(DATES)}"
    else:
        fact = f"我们的约定是周报{rng.choice(TIMES)}提交"
    head = rng.choice(["记一下：", "请记住：", "帮我记住，", "你记一下，"])
    obs = {"tool": "kb_add", "result": {"ok": True, "written": fact,
                                        "ms": round(rng.uniform(20, 90), 1),
                                        "kb_size": rng.randint(1, 400),
                                        "weights_unchanged": True,
                                        "note": "写进的是外部知识库文件，模型权重一个字节都没动"}}
    return _msgs(head + fact,
                 [(_call("kb_add", {"text": fact}), _obs(obs)), ("记下了。", None)]), "write"


def gen_calc(rng):
    a, b = rng.randint(12, 9999), rng.randint(11, 999)
    op, f = rng.choice([("+", lambda x, y: x + y), ("-", lambda x, y: x - y),
                        ("*", lambda x, y: x * y)])
    if op == "-" and a < b:
        a, b = b, a
    expr = f"{a} {op} {b}"
    v = f(a, b)
    q = rng.choice([f"{a} {op} {b} 等于多少？", f"帮我算 {expr}", f"{expr} 是多少？"])
    obs = {"tool": "calc", "result": {"ok": True, "expr": expr, "value": v}}
    return _msgs(q, [(_call("calc", {"expr": expr}), _obs(obs)),
                     (f"{expr} = {v}。", None)]), "calc"


def gen_none(rng):
    """本来就知道的：直接答，不许调工具。这是修「误触发 1/3」的主力。"""
    # 改写占七成：它的组合空间最大。平均分配的话，常识那几类几十条就用完了，
    # 整类会被去重挤掉（实测过：平均分配时这类只剩 366 条）
    which = rng.randrange(10)
    if which == 7:
        q, a = rng.choice(list(DEFINITIONS.items()))
    elif which == 8:
        q, a = rng.choice(CONVERTS)
        q += "？"
    elif which == 9:
        q = rng.choice(["你是谁？", "你能做什么？", "你能帮我做什么？", "你能干什么？"])
        a = rng.choice([
            "我是助手，可以回答问题、做精确计算，也可以把你的新事实写进外部知识库，以后帮你查回来。",
            "我是助手。能答题、能算数、能帮你记事情并查回来。",
            "我可以回答问题、精确计算，也能帮你把新事实存进知识库并在需要时查回来。"])
    else:
        # 改写：先有一句干净的，故意打错一个字，要的就是那句干净的。
        # 这里用组合式模板，不然整类只有几条，去重之后没剩多少（实测过）。
        field = rng.choice(["告警阈值", "超时时间", "重试次数", "连接池大小", "限流配额",
                            "缓存过期时间", "副本数", "分片数"])
        s = rng.choice(SYSTEMS)
        clean = rng.choice([
            f"{s}的{field}调到了 {rng.randint(1, 999)}",
            f"请{rng.choice(['周五', '下周一', '月底前', '明天'])}把{s}的"
            f"{rng.choice(['方案', '复盘', '排期', '设计文档'])}发我",
            f"{rng.choice(SURNAMES)}说{s}的{field}还要再调",
            f"上线前先回归一遍，{s}最近不稳定",
            f"记得在{rng.choice(['发布前', '灰度前', '切流量前'])}备份{s}的配置"])
        typo = clean.replace("的", "地", 1) if "的" in clean else clean.replace("了", "乐", 1)
        q = rng.choice([f"帮我把这句话改通顺：{typo}", f"这句话有错别字，改一下：{typo}",
                        f"把下面这句改顺：{typo}", f"校对一下：{typo}"])
        a = clean
    return _msgs(q, [(a, None)]), "none"


FAMILIES = [(gen_recall, 34), (gen_write, 22), (gen_none, 22), (gen_calc, 12),
            (gen_refuse, 10)]


def main() -> int:
    ap = argparse.ArgumentParser(description="造 SFT 数据（格式与推理时完全一致）")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default=str(ROOT.parent / "work-chat" / "sft" / "train.jsonl"))
    ap.add_argument("--val", default=str(ROOT.parent / "work-chat" / "sft" / "val.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    total_w = sum(w for _, w in FAMILIES)
    out, val = Path(args.out), Path(args.val)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, stats, tokens = set(), {}, 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_PATH)

    def ntok(msgs) -> int:
        # apply_chat_template(tokenize=True) 在 transformers 5 里返回的不一定是 token 列表，
        # 直接 len() 会算成键的个数。先把模板渲染成文本再编码，数出来才是 token 数。
        text = tok.apply_chat_template(msgs, tools=C.TOOLS, tokenize=False,
                                      add_generation_prompt=False)
        return len(tok(text)["input_ids"])

    val_rows = []
    with out.open("w", encoding="utf-8") as f:
        while len(seen) < args.n:
            r = rng.randrange(total_w)
            fn, acc = None, 0
            for g, w in FAMILIES:
                acc += w
                if r < acc:
                    fn = g
                    break
            msgs, kind = fn(rng)
            text = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
            if any(h in text for h in HOLDOUT):        # 评测集用过的实体不进训练
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            tokens += ntok(msgs)
            stats[kind] = stats.get(kind, 0) + 1
            if len(seen) <= 400:                        # 前 400 条留作验证集
                val_rows.append(text)
                continue
            f.write(text + "\n")
    with val.open("w", encoding="utf-8") as vf:
        vf.write("\n".join(val_rows) + "\n")
    print(f"生成 {len(seen)} 条（去重后）≈ {tokens:,} token；分布 {stats}")
    print(f"训练集 {out}（{len(seen) - len(val_rows)} 条）　验证集 {val}（{len(val_rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
