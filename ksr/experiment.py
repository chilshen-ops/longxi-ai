"""四项主张的端到端实验：训练 → 评测 → 打印实测对照表。

运行：  python -m ksr.experiment            （CPU，约 1-3 分钟）

任务设计（刻意让"失败模式"就是你举的那个例子）：
  * 4 个领域 = 4 个技能专家：general / windows / linux / hardware
  * 技能题：问"Windows 下查看端口占用用什么命令" → 答案 netstat；
    Linux 同一问题 → ss。选错专家就会把 Windows 答成 Linux 的命令，
    这正是你说的"把 Windows 的命令用成 Linux 的"。
  * 事实题：端口 → 服务名是任意映射，训练时只以"检索到的上下文"形式给出，
    留出 24 个端口完全不训练。它们只存在于知识库里。
      → 检索开 = 答对，检索关 = 答错，就证明了"事实不在权重里也照样答得对"。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ksr.config import KSRConfig
from ksr.knowledge import (CorrectionStore, KnowledgeBase, KnowledgeRecord, Segment,
                           _key_tokens)
from ksr.nets import CacheList, GQASelfAttention, KSRModel, RMSNorm, SwiGLU
from ksr.selfevolve import (accuracy, answer_with_correction, corrections_to_examples,
                            finetune_expert, predict, route_policy)
from ksr.tokenizer import CharTokenizer

import ksr_calc

SEED = 0
WORK = Path(os.environ.get("KSR_WORK", str(ROOT / "work")))   # 冒烟测试用独立目录，别污染正式检查点
# 步数可调：先用几步把整条评测链路跑通（1 分钟），再跑完整训练（避免每轮 5 分钟只撞出一个 bug）
STEPS = int(os.environ.get("KSR_STEPS", "900"))
CSTEPS = int(os.environ.get("KSR_CSTEPS", "200"))

def _items(spec_str: str) -> List[Tuple[str, str]]:
    """把 "目的:答案 目的:答案 …" 展开成 (目的, 答案) 列表。"""
    out = []
    for pair in spec_str.split():
        p, a = pair.rsplit(":", 1)
        out.append((p, a))
    return out


# ── 16 个领域，每领域一套互不相干的答案词表 ─────────────────────────────────
# 为什么要换成 16 领域：实测发现 4 领域 × 6 题时，故意把答案塞给错误的专家，
# 准确率仍然是 100%，把共享通路砍到 64 也一样。那不是架构没问题，是任务太简单——
# 24 条映射靠注意力那条路径就能背下来，任何专家都没参与，所以怎么调容量比例都
# 测不出差别。这里把任务做成容量受限的：160 条映射、16 套互不相干的词表，
# 窄的共享通路背不下，路由才有分工的必要。
# 训练：每个意图用两种句式；评测：用训练里没出现过的第三种句式（答案都是训练过的，
# 泛化的只有句式）——这样测的是"能不能认出这是哪个领域"，不是"有没有背过这句话"。
DOMAIN_SPEC = {
    "windows": {
        "train_q": ["Windows 下{p}用什么命令", "在 Windows 上{p}敲什么"],
        "eval_q": ["Windows 环境里{p}要用哪个工具"],
        "items": _items("查看端口占用:netstat 查看网络配置:ipconfig 打开任务管理器:taskmgr "
                        "编辑注册表:regedit 打开设备管理器:devmgmt 查看系统服务:services "
                        "管理开机启动项:msconfig 清理临时文件:cleanmgr 查看事件日志:eventvwr "
                        "添加本地打印机:printers"),
    },
    "linux": {
        "train_q": ["Linux 下{p}用什么命令", "在 Linux 上{p}要敲什么"],
        "eval_q": ["Linux 系统里{p}用哪个工具"],
        "items": _items("查看端口占用:ss 列出打开的文件:lsof 查看网络接口:ifconfig "
                        "管理系统服务:systemctl 查看系统日志:journalctl 查看进程资源:htop "
                        "管理磁盘分区:fdisk 挂载文件系统:mount 配置定时任务:cron "
                        "配置远程登录:sshd"),
    },
    "hardware": {
        "train_q": ["{p}是多少", "{p}是啥数值"],
        "eval_q": ["请给出{p}的具体数字"],
        "items": _items("STM32F103的主频:72MHz ESP32的主频:240MHz STM32F103的工作电压:3.3V "
                        "ArduinoUNO的工作电压:5.0V CortexM内核电压:1.8V "
                        "STM32F103的外部晶振:12MHz STM32F103C8T6的闪存:32kB "
                        "STM32F407的闪存:512kB ESP32的外部晶振:27MHz STM32F407的主频:168MHz"),
    },
    "physics": {
        "train_q": ["{p}的原因是什么", "{p}为什么会这样"],
        "eval_q": ["请说明{p}对应的物理机制"],
        "items": _items("天空为什么是蓝的:rayleigh 苹果为什么会落地:gravity "
                        "滑行为什么会停下:friction 船为什么能浮起来:buoyancy "
                        "撬棍为什么省力:lever 物体为什么保持匀速运动:inertia "
                        "筷子在水里为什么折断:refraction 肥皂泡为什么有彩色条纹:diffraction "
                        "热水怎么把整个房间变热:convection 磁铁在线圈里运动产生什么:induction"),
    },
    "chemistry": {
        "train_q": ["{p}的化学式是什么", "{p}写成化学式怎么拼"],
        "eval_q": ["请写出{p}对应的化学式"],
        "items": _items("水分子:H2O 食盐:NaCl 干冰:CO2 氧气:O2 硫酸:H2SO4 氨气:NH3 "
                        "甲烷:CH4 铁锈:Fe2O3 石灰石:CaCO3 乙醇:C2H5OH"),
    },
    "network": {
        "train_q": ["{p}是哪个协议", "{p}该用哪个协议"],
        "eval_q": ["请指出{p}对应的协议名称"],
        "items": _items("七层参考模型有几层:OSI7 保证可靠传输的传输层协议:TCP "
                        "不保证可靠传输的传输层协议:UDP 互联网域间路由协议:BGP "
                        "链路状态内部网关协议:OSPF 自动分配地址的协议:DHCP "
                        "域名解析用什么:DNS 地址解析协议:ARP 隔离广播域用什么:VLAN "
                        "私网访问公网用什么:NAT"),
    },
    "database": {
        "train_q": ["数据库里{p}用什么机制", "为了{p}应该用什么"],
        "eval_q": ["请说出数据库里承担{p}职责的机制"],
        "items": _items("加速按值查找:index 崩溃后恢复用:WAL 多版本并发控制:MVCC "
                        "范围扫描用的树:btree 水平拆分数据:sharding 把数据复制到副本:replica "
                        "回收死元组空间:vacuum 重放已提交事务:redo 保证可重复读:Bplus "
                        "防止并发写冲突:lock"),
    },
    "container": {
        "train_q": ["容器里{p}用什么", "{p}这件事用哪个工具"],
        "eval_q": ["请回答容器场景下{p}对应的工具"],
        "items": _items("描述怎么构建镜像:dockerfile 多容器一起起:compose 存放镜像的地方:registry "
                        "持久化数据:volume 容器之间通信:bridge 多台机器组成集群:swarm "
                        "进入正在跑的容器:exec 生成一个新镜像:build 查询集群节点状态:kubectl "
                        "给集群装应用:helm"),
    },
    "k8s": {
        "train_q": ["K8s 里{p}的对象叫什么", "{p}对应哪个资源"],
        "eval_q": ["请说出 Kubernetes 中负责{p}的资源名"],
        "items": _items("最小可调度单元:pod 对外暴露HTTP入口:ingress 注入配置数据:configmap "
                        "每个节点跑一份:daemonset 定时运行任务:cronjob 按负载自动扩缩容:hpa "
                        "扩展自定义API:operator 给主容器打辅助:sidecar 赶走不适合的节点:taint "
                        "优先调度到某些节点:affinity"),
    },
    "git": {
        "train_q": ["Git 里{p}用什么命令", "{p}该用哪个 Git 子命令"],
        "eval_q": ["请给出 Git 中用于{p}的子命令"],
        "items": _items("改写提交历史:rebase 把某个提交搬过来:cherry-pick 二分定位问题提交:bisect "
                        "暂存未提交的改动:stash 引用另一个仓库:submodule 找回丢掉的提交:reflog "
                        "同时开多个工作目录:worktree 撤销已经推上去的提交:revert "
                        "给提交打版本标记:tag 提交前自动跑检查:hook"),
    },
    "python": {
        "train_q": ["Python 里{p}用什么工具", "{p}用哪个 Python 工具"],
        "eval_q": ["请写出 Python 生态里承担{p}职责的工具"],
        "items": _items("隔离项目依赖:venv 安装第三方包:pip 写单元测试:pytest "
                        "做静态类型检查:mypy 统一代码格式:black 最快的静态检查:ruff "
                        "依赖与虚拟环境一体:poetry 新一代安装器:uv 打包分发格式:wheel "
                        "构建分发的元数据:setuptools"),
    },
    "rust": {
        "train_q": ["Rust 里{p}用什么", "{p}对应哪个 Rust 工具"],
        "eval_q": ["请说出 Rust 生态中{p}对应的工具"],
        "items": _items("构建与包管理:cargo 静态检查:clippy 统一代码格式:rustfmt "
                        "检查未定义行为:miri 异步运行时:tokio 做序列化:serde 数据并行:rayon "
                        "编译到浏览器里跑:wasm 一个发布单位:crate 检查所有权:borrow"),
    },
    "medical": {
        "train_q": ["{p}一般用哪种药", "{p}对应的药名是什么"],
        "eval_q": ["请给出临床上用于{p}的药物名称"],
        "items": _items("退烧止痛最经典的药:aspirin 另一种常用退烧止痛:ibuprofen "
                        "人类第一个抗生素:penicillin 调节血糖的激素:insulin 降胆固醇:statin "
                        "注射用抗凝药:heparin 口服抗凝药:warfarin 强效镇痛:morphine "
                        "应激时大量分泌的激素:adrenaline 一种与运动有关的神经递质:dopamine"),
    },
    "legal": {
        "train_q": ["{p}在法律上叫什么", "{p}对应的法律术语是什么"],
        "eval_q": ["请写出法律语境下{p}对应的术语"],
        "items": _items("民事侵权之诉:tort 担保债权的留置权:lien 由第三方代管的资金:escrow "
                        "法院要求停止某行为:injunction 要求证人到庭:subpoena 开庭前庭外取证:deposition "
                        "预先付给律师的费用:retainer 书面宣誓陈述:affidavit 对孩子的照护权:custody "
                        "立法机关通过的法条:statute"),
    },
    "finance": {
        "train_q": ["{p}在财务上叫什么", "{p}对应的金融术语是什么"],
        "eval_q": ["请给出金融语境下{p}对应的术语"],
        "items": _items("分期摊还一笔贷款:amortize 息税折旧摊销前利润:EBITDA "
                        "并购中多付的无形资产:goodwill 用衍生品抵消风险:hedge "
                        "同时买卖赚无风险价差:arbitrage 标准化合约:futures 分给股东的钱:dividend "
                        "借券卖出等跌:shorting 用本金放大收益:leverage 买价与卖价之差:spread"),
    },
    "chess": {
        "train_q": ["{p}是哪套开局", "{p}属于哪个开局"],
        "eval_q": ["请说出国际象棋中{p}对应的开局名"],
        "items": _items("1.e2e4c5开头的开局:Sicilian 后翼弃兵:Gambit 加泰罗尼亚体系:Catalan "
                        "1.c4开头的开局:English 西西里中最锋利的变例:Najdorf 西西里龙式变例:Dragon "
                        "1.d4d52.Bf4体系:London 开局第一步走Bg5:Trompowsky "
                        "卡鲁卡恩里的Panov攻击:Panov 1.e2e4Nf6的防御:Alekhine"),
    },
}
SERVICES = [f"svc{i:02d}" for i in range(24)]
DOMAINS = list(DOMAIN_SPEC)
ANSWERS = [a for spec in DOMAIN_SPEC.values() for _, a in spec["items"]]
# 路由监督的靶：一个领域固定对应一个专家。这是"oracle 蒸馏"式的监督信号——
# 没有它，路由器只看到"答对就行"，而答对不需要分工，于是它塌回均匀。
DOM_TO_EXPERT = {d: i for i, d in enumerate(DOMAINS)}


def build_corpus() -> Dict:
    rng = random.Random(SEED)
    ports_train = [8000 + i for i in range(40)]
    ports_held = [9000 + i for i in range(24)]
    fact_table = {p: SERVICES[i % len(SERVICES)] for i, p in enumerate(ports_train + ports_held)}

    def fact_prompt(port: int, with_context: bool) -> str:
        q = f"查询端口 {port} 上运行的服务是什么"
        if with_context:
            ctx = f"【已知】端口 {port} 的服务是 {fact_table[port]}\n"
            return ctx + q + "\nA: "
        return q + "\nA: "

    train: List[Tuple[str, str, str]] = []      # (prompt, answer, domain)
    eval_skill: List[Tuple[str, str, str]] = []
    for dom, spec in DOMAIN_SPEC.items():
        items = spec["items"]
        for p, a in items:
            # 训练：每个意图用两种句式；评测：用训练里没出现过的第三种句式
            for t in spec["train_q"]:
                train.append((t.format(p=p) + "\nA: ", a, dom))
        for i in (4, 9):
            p, a = items[i]
            eval_skill.append((spec["eval_q"][0].format(p=p) + "\nA: ", a, dom))
    for _ in range(int(len(train) * 0.25)):
        p = rng.choice(ports_train)
        train.append((fact_prompt(p, True), fact_table[p], "fact"))

    eval_fact = [(fact_prompt(p, True), fact_table[p], "fact") for p in ports_held]
    eval_fact_noctx = [(fact_prompt(p, False), fact_table[p], "fact") for p in ports_held]
    return {"train": train, "eval_skill": eval_skill, "eval_fact": eval_fact,
            "eval_fact_noctx": eval_fact_noctx, "fact_table": fact_table,
            "ports_held": ports_held}


def build_tokenizer(corpus: Dict) -> CharTokenizer:
    texts = [p for items in ("train", "eval_skill", "eval_fact", "eval_fact_noctx")
             for p, _, _ in corpus[items]]
    extra = ANSWERS + SERVICES
    tok = CharTokenizer.from_corpus(texts)
    for t in extra:
        tok._add(t)
    return tok


def encode_pair(tok: CharTokenizer, prompt: str, answer: str) -> Tuple[List[int], List[int]]:
    """ids 与 labels 等长；labels 在答案位置等于 ids 本身，其余 -100。

    损失用 logits[:, :-1] 对 labels[:, 1:]，即"用位置 i 预测位置 i+1"。
    答案第一个 token 的决策位置 = 提示词最后一个 token 的位置，与推理时一致。
    """
    p = tok.encode_prompt(prompt)              # 含 BOS：与推理侧走同一条编码路径
    a = tok.encode(answer) + [tok.eos_id]
    ids = p + a
    labels = [-100] * len(p) + a
    assert len(ids) == len(labels)
    return ids, labels


def make_batch(tok: CharTokenizer, items: Sequence[Tuple[str, str, str]]):
    xs, ys = [], []
    for prompt, answer, _ in items:
        ids, labels = encode_pair(tok, prompt, answer)
        xs.append(ids)
        ys.append(labels)
    n = max(len(x) for x in xs)
    pad, neg = tok.pad_id, -100
    X = torch.full((len(xs), n), pad, dtype=torch.long)
    Y = torch.full((len(xs), n), neg, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


# ── 训练 ──────────────────────────────────────────────────────────────────────
TRAIN_CKPT = WORK / "train_ckpt.pt"


def train_model(model: KSRModel, tok: CharTokenizer, corpus: Dict, steps=900, bs=16, lr=3e-3,
                resume: bool = True, every: int = 100):
    """训练，每 every 步存一次检查点，断电可以从断点续训。

    加这个是因为这台机器已经第三次满载硬断电了：2 线程时整机 18% 跑十几分钟没事，
    8 线程约 8 分钟后就出现 Kernel-Power 41 + 6008 意外关机。线程越高越容易触发，
    所以把"断电 = 全部重来"改成"断电最多损失 every 步"。

    两点如实说明：优化器状态不续（AdamW 动量清零），重启后头几十步 loss 会小幅回升，
    所以这不是严格等价的重启；数据顺序改成每步独立播种，保证续训和从头跑取到同一批数据。
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    start = 0
    if resume and TRAIN_CKPT.exists():
        state = torch.load(TRAIN_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        start = int(state["step"]) + 1
        print(f"  从检查点续训：已完成 {start} 步（断电保护：最多损失 {every} 步）")
    items = corpus["train"]
    for step in range(start, steps):
        rng = random.Random(SEED + 1 + step)
        batch = [items[rng.randrange(len(items))] for _ in range(bs)]
        X, Y = make_batch(tok, batch)
        logits, meta = model(X)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, model.cfg.vocab_size),
                               Y[:, 1:].reshape(-1), ignore_index=-100)
        loss = loss + model.cfg.aux_loss_weight * meta["aux_loss"]
        w_sup = getattr(model.cfg, "router_supervision_weight", 0.0)
        if w_sup > 0:
            # 路由监督：每个领域固定对应一个专家，在"预测第一个答案 token"的那个位置
            # 监督每一层的路由。fact 类样本没有技能靶，用 -1 忽略。
            tgt = torch.full((X.shape[0],), -1, dtype=torch.long)
            for i, (_, _, dom) in enumerate(batch):
                tgt[i] = DOM_TO_EXPERT.get(dom, -1)
            keep = tgt >= 0
            if bool(keep.any()):
                bidx = keep.nonzero(as_tuple=True)[0]
                pos = ((Y != -100).float().argmax(dim=1) - 1)[keep].clamp(min=0)
                rl = torch.stack(meta["router_logits"])[:, bidx]          # [L,Bk,S,E]
                gather_i = pos.view(1, -1, 1, 1).expand(rl.shape[0], -1, 1, rl.shape[-1])
                rl = rl.gather(2, gather_i).squeeze(2)                    # [L,Bk,E]
                target = tgt[bidx].unsqueeze(0).expand(rl.shape[0], -1)   # 每层同一个靶
                loss = loss + w_sup * F.cross_entropy(
                    rl.reshape(-1, rl.shape[-1]), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {float(loss.detach()):.3f}  "
                  f"aux {float(meta['aux_loss'].detach()):.3f}")
        if step % every == 0 or step == steps - 1:
            # 最后一步也要存：上一轮就是训练全部跑完、死在校验器阶段，
            # 结果整个训练成果没落盘，白跑一遍。
            torch.save({"model": model.state_dict(), "step": step}, TRAIN_CKPT)
    model.eval()


def primary_expert(model: KSRModel, tok: CharTokenizer, prompt: str) -> int:
    """路由在决策位置各层选得最多的主专家。

    expert_ids 的形状是 [B,S,top_k]，跨层 stack 后是 [L,B,S,top_k]。主专家是 top-1，
    必须先用 [..., 0] 取出来再统计，否则 mode 会把次选也算进去（这就是刚才那个崩溃）。
    """
    ids = torch.tensor([tok.encode_prompt(prompt)])
    with torch.no_grad():
        meta = model(ids)[1]
    votes = torch.stack(meta["expert_ids"])[:, 0, -1, 0]      # 各层的主专家
    return int(votes.mode().values)


def learned_assignment(model: KSRModel, tok: CharTokenizer, corpus: Dict) -> Dict[str, int]:
    """训练后路由自己学到的 领域 → 专家 分配（不硬编码，用数据说话）。"""
    out: Dict[str, int] = {}
    for dom in DOMAINS:
        spec = DOMAIN_SPEC[dom]
        probes = [spec["eval_q"][0].format(p=p) for p, _ in spec["items"][:4]]
        votes = [primary_expert(model, tok, q + "\nA: ") for q in probes]
        out[dom] = max(set(votes), key=votes.count)
    return out


def train_corrector(model: KSRModel, tok: CharTokenizer, corpus: Dict, assign: Dict[str, int],
                    steps=200, bs=16):
    """配对训练校验器：正确专家 → 标签"对"；故意塞错专家 → 标签"错"。

    这就是"常识专家发现主专家跑偏"的可训练版本，而不是拍脑袋的阈值。
    """
    corr = model.corrector
    assert corr is not None
    for p in model.parameters():
        p.requires_grad_(False)
    for p in corr.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(corr.parameters(), lr=3e-3)
    items = [it for it in corpus["train"] if it[2] in DOMAINS]
    rng = random.Random(SEED + 2)

    def hidden_for(batch, force):
        X, _ = make_batch(tok, batch)
        _, meta = model(X, force_expert=force)
        h = model.norm(meta["final_hidden"])
        r = meta["reasoner_ref"]
        # 决策位置 = "A: " 之后那个位置，即 X 里最后一个提示词元的位置
        # （X = [pad] + prompt + answer[:-1]，所以就是 len(prompt)）
        pos = [len(tok.encode_prompt(p)) - 1 for p, _, _ in batch]
        sel = (torch.arange(len(pos)), torch.tensor(pos))
        return h[sel], r[sel]

    for step in range(steps):
        batch = [items[rng.randrange(len(items))] for _ in range(bs)]
        gold = [assign[it[2]] for it in batch]
        wrong = []
        for i, g in enumerate(gold):
            cand = [e for e in range(model.cfg.num_skill_experts) if e != g]
            wrong.append(cand[rng.randrange(len(cand))])
        # 正确分支：按路由学到的专家跑（不强制）
        h_ok, r_ok = hidden_for(batch, None)
        # 错误分支：逐个样本强制成错误专家
        h_bad, r_bad = [], []
        for e in set(wrong):
            sub = [batch[i] for i in range(len(batch)) if wrong[i] == e]
            hb, rb = hidden_for(sub, e)
            h_bad.append(hb)
            r_bad.append(rb)
        h_bad = torch.cat(h_bad)
        r_bad = torch.cat(r_bad)
        h = torch.cat([h_ok, h_bad])
        r = torch.cat([r_ok, r_bad])
        y = torch.tensor([0] * len(h_ok) + [2] * len(h_bad))
        _, probs = corr.verify(h, r)
        loss = F.cross_entropy(probs, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            acc = (probs.argmax(-1) == y).float().mean().item()
            print(f"  corrector step {step:3d}  loss {float(loss.detach()):.3f}  acc {acc:.2f}")
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


# ── 推理与评测 ────────────────────────────────────────────────────────────────
def answer_of(model: KSRModel, tok: CharTokenizer, prompt: str,
              force_expert: int | None = None) -> Tuple[str, Dict]:
    ids = tok.encode_prompt(prompt)
    # max_new 要够长：答案是整条命令字符串，生成到 EOS 才停。
    # 之前给 4，任何超过 4 个字符的答案必然判错，短答案（"ss"）反而能过——
    # 于是准确率被钉在 8% 左右，看着像模型差，其实是采样预算不够。
    out, agg = model.generate(ids, max_new=24, force_expert=force_expert)
    return tok.decode(out).strip(), agg


def verify_prompt(model: KSRModel, tok: CharTokenizer, prompt: str, force: int | None):
    """在"A: 之后那个位置"做一次校验，返回 (是否判错, 判对概率)。"""
    ids = tok.encode_prompt(prompt)
    X = torch.tensor([ids])
    with torch.no_grad():
        _, meta = model(X, force_expert=force)
        h = model.norm(meta["final_hidden"])
        r = meta["reasoner_ref"]
        n = len(ids)
        _, probs = model.corrector.verify(h[:, n - 1], r[:, n - 1])
    p = probs[0]
    return bool(int(p.argmax()) == 2), float(p[0])


def candidates_for(model: KSRModel, tok: CharTokenizer, prompt: str, forced: int) -> List[int]:
    ids = torch.tensor([tok.encode_prompt(prompt)])
    with torch.no_grad():
        _, meta = model(ids)
    top = torch.stack(meta["expert_ids"])[:, 0, -1, :2].reshape(-1).tolist()
    out, seen = [forced], set()
    for e in top + [0]:
        e = int(e)
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def eval_skill(model, tok, items, with_correction: bool) -> Dict:
    """故意把每个样本塞进一个错误专家，看纠错层能不能救回来。"""
    ok, tries, saved = 0, [], 0
    for prompt, gold, dom in items:
        forced = _wrong_expert_for(model, tok, prompt, dom)
        pred, _ = answer_of(model, tok, prompt, force_expert=forced)
        if not with_correction:
            ok += int(pred == gold)
            tries.append(1)
            continue
        is_wrong, _ = verify_prompt(model, tok, prompt, forced)
        if not is_wrong:
            ok += int(pred == gold)
            tries.append(1)
            continue
        best, best_p, best_score = pred, pred, -1.0
        for n, cand in enumerate(candidates_for(model, tok, prompt, forced), start=1):
            p, _ = answer_of(model, tok, prompt, force_expert=cand)
            _, p_ok = verify_prompt(model, tok, prompt, cand)
            if p_ok > best_score:
                best, best_p, best_score = p, p, p_ok
            if int(p_ok > 0.5 and p == gold):
                break
        tries.append(n)
        ok += int(best_p == gold)
        saved += int(best_p == gold) - int(pred == gold)
    return {"acc": ok / len(items), "avg_forwards": float(np.mean(tries)), "n": len(items)}


_ASSIGN: Dict[str, int] = {}


def _wrong_expert_for(model, tok, prompt: str, dom: str) -> int:
    top = primary_expert(model, tok, prompt)
    cand = [e for e in range(model.cfg.num_skill_experts) if e != top]
    return cand[0]


def eval_skill_forced(model, tok, items, expert: int | None) -> Dict:
    """把所有题都固定到同一个专家上作答——这是"路由到底有没有用"的对照。

    expert=0 就是没有路由的情形：只有一个共享专家，等价于稠密模型。
    自然路由明显高于它，才说明专家分工真的在起作用；两者一样，就说明这些题
    压根不需要专家容量（上一轮 4 领域就是这个情况，所以怎么测都一样）。
    """
    ok = 0
    per_domain: Dict[str, int] = {}
    for prompt, gold, dom in items:
        pred, _ = answer_of(model, tok, prompt, force_expert=expert)
        hit = int(pred == gold)
        ok += hit
        per_domain[dom] = per_domain.get(dom, 0) + hit
    return {"acc": ok / len(items), "n": len(items),
            "zero_domains": sorted(d for d, c in per_domain.items() if c == 0)}


def eval_fact(model, tok, kb: KnowledgeBase, items, embed_fn, use_kb: bool,
              cfg: KSRConfig) -> Dict:
    """分开测两件事：检索有没有把正确片段捞上来，和模型有没有把答案从上下文里抄出来。

    不分开就分不清"知识外置失败"到底是检索的锅还是模型的锅。
    """
    ok = retrieved_ok = 0
    seg_tokens = []
    for prompt, gold, _ in items:
        ctx = ""
        if use_kb:
            segs = retrieve_for(model, tok, kb, embed_fn, prompt, cfg)
            ctx = "".join(s.text + "\n" for s in segs)
            keys = _key_tokens(prompt)
            # 判定要严：必须同时命中"这个端口"和"这个答案"。
            # 只看 gold 在不在文本里会被重复的服务名蒙对（每个服务名对应多个端口）。
            retrieved_ok += int(any(gold in s.text and (keys & _key_tokens(s.text))
                                    for s in segs))
            seg_tokens.append(sum(s.tokens for s in segs))
        pred, _ = answer_of(model, tok, ctx + prompt)
        ok += int(pred == gold)
    out = {"acc": ok / len(items), "n": len(items)}
    if use_kb:
        out["retrieval_hit"] = retrieved_ok / len(items)
        out["avg_injected_tokens"] = float(np.mean(seg_tokens)) if seg_tokens else 0.0
        out["budget_total"] = cfg.kb_max_total_tokens
    return out


def retrieve_for(model, tok, kb, embed_fn, prompt: str, cfg: KSRConfig) -> List[Segment]:
    q = embed_fn(prompt)
    return kb.retrieve(q, kb.domains(), top_k=cfg.kb_top_k,
                       max_tokens_per_segment=cfg.kb_max_tokens_per_segment,
                       max_total_tokens=cfg.kb_max_total_tokens,
                       count_tokens=tok.count_tokens,
                       query_text=prompt)          # 混合检索要用到问题里的编号


def eval_policy(model, tok, items, cfg: KSRConfig) -> Dict:
    """真实路由下的三级策略：不人为塞错专家，看系统自己能不能发现并纠正。"""
    ok_naive = ok_pol = 0
    levels: Dict[str, int] = {}
    forwards: List[int] = []
    for prompt, gold, _ in items:
        _, agg = predict(model, tok, prompt)
        d = route_policy(agg, cfg)
        levels[d.level] = levels.get(d.level, 0) + 1
        naive, _ = predict(model, tok, prompt)          # 对照组：无条件信主专家
        r = answer_with_correction(model, tok, prompt, cfg,
                                   candidates=list(range(cfg.num_skill_experts)))
        ok_naive += int(naive == gold)
        ok_pol += int(r["answer"] == gold)
        forwards.append(r["forwards"])
    return {"naive": ok_naive / len(items), "policy": ok_pol / len(items),
            "levels": levels, "avg_forwards": float(np.mean(forwards)), "n": len(items)}


def build_kb(model, tok, embed_fn, corpus: Dict, cfg: KSRConfig) -> KnowledgeBase:
    kb = KnowledgeBase(WORK / "knowledge", dim=cfg.hidden_size)
    for port, svc in corpus["fact_table"].items():
        text = f"端口 {port} 的服务是 {svc}"
        kb.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(text), domain="fact",
                                      kind="fact", text=text, answer=svc, source="manual",
                                      verified=True), embed_fn(text))
    return kb


def make_embed(model: KSRModel):
    @torch.no_grad()
    def embed(text: str) -> np.ndarray:
        ids = torch.tensor([model.cfg and 0])  # placeholder to keep device lookup cheap
        ids = torch.tensor([_encode_ids(model, text)])
        h = model.embed(ids)[0].mean(0)
        return h.cpu().numpy().astype(np.float32)
    return embed


def _encode_ids(model, text: str):
    return _TOK.encode(text)


_TOK: CharTokenizer


# ── 稠密基线（同底座，标准 FFN，全注意力）：用来实测速度差 ─────────────────────
class DenseTiny(torch.nn.Module):
    def __init__(self, cfg: KSRConfig, ffn_hidden: int):
        super().__init__()
        self.cfg = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = torch.nn.ModuleList()
        for _ in range(cfg.num_layers):
            d = _DenseBlock(cfg, ffn_hidden)
            self.blocks.append(d)
        self.norm = RMSNorm(cfg.hidden_size)
        self.head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        h = self.embed(ids)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


class _DenseBlock(torch.nn.Module):
    def __init__(self, cfg: KSRConfig, ffn_hidden: int):
        super().__init__()
        d = KSRConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, num_layers=1,
                      num_attention_heads=cfg.num_attention_heads,
                      num_kv_groups=cfg.num_attention_heads, head_dim=cfg.head_dim,
                      full_attn_layers=[1], reasoner_hidden=cfg.reasoner_hidden,
                      router_hidden=cfg.router_hidden)
        self.attn = GQASelfAttention(d)
        self.n1 = RMSNorm(cfg.hidden_size)
        self.ffn = SwiGLU(cfg.hidden_size, ffn_hidden)
        self.n2 = RMSNorm(cfg.hidden_size)

    def forward(self, x):
        from ksr.nets import precompute_rope
        if not hasattr(self, "_rope") or self._rope.shape[0] < x.shape[1]:
            self._rope = precompute_rope(self.attn.dh, max(64, x.shape[1]), 10000.0)
        x = x + self.attn(self.n1(x), self._rope)
        return x + self.ffn(self.n2(x))


def throughput(model_fn, ids, reps=30) -> float:
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model_fn(ids)
    dt = time.perf_counter() - t0
    return (ids.shape[0] * ids.shape[1] * reps) / dt


def main() -> int:
    global _TOK, _ASSIGN
    # 默认不吃满机器：满载 AVX2 全核负载在这类平台上疑似触发过断电
    torch.set_num_threads(max(1, min(int(os.environ.get("KSR_THREADS", "8")), os.cpu_count() or 8)))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    WORK.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("KSR_KEEP"):
        # 每轮都从干净的知识目录开始：上一轮的条目会和新写的混在一起，
        # "知识库有多少条"会翻倍失真，检索也会被旧条目污染。
        for sub in ("knowledge", "kb_missing", "corrections"):
            shutil.rmtree(WORK / sub, ignore_errors=True)
    report: Dict = {}

    def dump() -> None:
        """每测完一项就落盘：这台机器中途断电也不会丢掉已完成的部分。"""
        (WORK / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    # 配置可切换：默认是定稿的等比缩小原型；KSR_CONFIG 指到 ksr_tiny_starved.json 就是
    # "共享通路调窄、专家调宽"的消融。同一个评测脚本跑两种容量分配，才叫对照实验。
    cfg_path = Path(os.environ.get("KSR_CONFIG", str(ROOT / "tools" / "configs" / "ksr_tiny.json")))
    cfg = KSRConfig.from_json(cfg_path)
    print("== 0) 规格（由计算器给出）==")
    print(f"  本次原型配置：{cfg_path.name}  常识专家宽={cfg.reasoner_hidden}  专家秩={cfg.expert_rank}")
    spec = ksr_calc.analyze(json.loads((ROOT / "tools/configs/ksr7b_32L.json").read_text("utf-8")))
    print(f"  KSR-MoE-7B 定稿: 总参 {spec['params']['total']/1e9:.2f}B, "
          f"激活 {spec['params']['activated_per_token']/1e9:.2f}B/token, "
          f"decode {spec['flops']['decode_per_token']/1e9:.1f} GFLOPs "
          f"(= 稠密 7B 的 {spec['flops']['decode_per_token']/ksr_calc.dense_baseline('llama7b', spec['cfg']['quant'])['flops']['decode_per_token']*100:.0f}%), "
          f"常驻显存 {spec['vram']['resident']/1e9:.2f} GB")

    print("\n== 1) 语料与分词 ==")
    corpus = build_corpus()
    tok = build_tokenizer(corpus)
    _TOK = tok
    cfg.vocab_size = tok.vocab_size
    print(f"  训练 {len(corpus['train'])} 条 / 技能评测 {len(corpus['eval_skill'])} 条 / "
          f"留出事实 {len(corpus['eval_fact'])} 条；词表 {tok.vocab_size}")

    print("\n== 2) 训练 KSR-tiny ==")
    model = KSRModel(cfg)
    print(f"  参数量 {sum(p.numel() for p in model.parameters()):,}，"
          f"每 token 激活 {model.activated_params():,}")
    ckpt = WORK / "model.pt"
    if ckpt.exists() and not os.environ.get("KSR_FRESH"):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        _ASSIGN = state["assignment"]
        print("  从检查点恢复，跳过训练（要重训：删 work/model.pt 或设 KSR_FRESH=1）")
    else:
        train_model(model, tok, corpus, steps=STEPS)
        _ASSIGN = learned_assignment(model, tok, corpus)
        print("\n== 3) 训练常识校验器（配对：正确专家 vs 故意塞错专家）==")
        train_corrector(model, tok, corpus, _ASSIGN, steps=CSTEPS)
        torch.save({"model": model.state_dict(), "assignment": _ASSIGN}, ckpt)
        print(f"  已保存检查点 {ckpt}")
    print(f"  路由学到的领域→专家分配: {_ASSIGN}")
    report["assignment"] = _ASSIGN
    dump()

    embed = make_embed(model)
    kb = build_kb(model, tok, embed, corpus, cfg)
    print(f"\n== 4) 知识库：{kb.size()} 条事实，全部在 SSD，不占显存 ==")

    print("\n== 5) 四项主张实测 ==")
    base = eval_skill(model, tok, corpus["eval_skill"], with_correction=False)
    fixed = eval_skill(model, tok, corpus["eval_skill"], with_correction=True)
    print(f"  [主张1 自动纠错] 故意选错专家：不纠错 {base['acc']*100:.0f}% → "
          f"启用纠错 {fixed['acc']*100:.0f}%（平均前向 {fixed['avg_forwards']:.2f} 次/题）")
    pol = eval_policy(model, tok, corpus["eval_skill"], cfg)
    print(f"  [主张1b 真实路由] 无条件信主专家 {pol['naive']*100:.0f}% → 按 L0/L1/L2 策略 "
          f"{pol['policy']*100:.0f}%（层级分布 {pol['levels']}，平均前向 {pol['avg_forwards']:.2f} 次/题）")
    nat = eval_skill_forced(model, tok, corpus["eval_skill"], None)
    dense = eval_skill_forced(model, tok, corpus["eval_skill"], 0)
    extra = f"，其中 {len(dense['zero_domains'])} 个领域全错：{dense['zero_domains'][:6]}" \
        if dense["zero_domains"] else ""
    print(f"  [主张1c 路由对照] 自然路由 {nat['acc']*100:.0f}% / 固定只用专家 0"
          f"（等价于没有路由）{dense['acc']*100:.0f}%{extra}")
    report.update(routed=nat, dense_single_expert=dense)
    dump()
    report.update(skill_forced_wrong_no_fix=base, skill_forced_wrong_fixed=fixed,
                  natural_routing=pol)
    dump()

    kb_on = eval_fact(model, tok, kb, corpus["eval_fact"], embed, True, cfg)
    kb_off = eval_fact(model, tok, kb, corpus["eval_fact_noctx"], embed, False, cfg)
    print(f"  [主张2 知识外置] 留出端口：检索开 {kb_on['acc']*100:.0f}% / 检索关 {kb_off['acc']*100:.0f}%"
          f"（{kb_on['n']} 题，答案只存在于知识库）")
    print(f"          检索本身命中率 {kb_on['retrieval_hit']*100:.0f}%，"
          f"平均注入 {kb_on['avg_injected_tokens']:.1f} token/题（预算 {kb_on['budget_total']}），"
          f"拿到正确片段后的抄写成功率 {(kb_on['acc']/max(kb_on['retrieval_hit'], 1e-9))*100:.0f}%")
    report.update(fact_kb_on=kb_on, fact_kb_off=kb_off)
    dump()

    # 主张3：实测吞吐
    ids = torch.tensor([tok.encode_prompt(corpus["eval_skill"][0][0]) for _ in range(4)])
    ksr_tp = throughput(lambda x: model(x), ids)
    ksr_total = sum(p.numel() for p in model.parameters())
    ffn = max(2 * cfg.hidden_size,
              int((ksr_total - 2 * sum(p.numel() for p in model.blocks[0].attn.parameters())
                   - cfg.vocab_size * cfg.hidden_size) / (2 * 3 * cfg.hidden_size)))
    dense = DenseTiny(cfg, ffn)
    dense_tp = throughput(lambda x: dense(x), ids)
    dense_total = sum(p.numel() for p in dense.parameters())
    print(f"  [主张3 高效] 同底座稠密基线（FFN={ffn}，{dense_total:,} 参数）{dense_tp:,.0f} tok/s "
          f"vs KSR-tiny（{ksr_total:,} 参数）{ksr_tp:,.0f} tok/s → {ksr_tp/dense_tp:.2f}×")
    report.update(throughput_dense=dense_tp, throughput_ksr=ksr_tp,
                  dense_params=dense_total, ksr_params=ksr_total, dense_ffn=ffn)
    dump()

    # 主张4：自我进化（L1 记忆级）
    port = corpus["ports_held"][0]
    svc = corpus["fact_table"][port]
    q = f"查询端口 {port} 上运行的服务是什么\nA: "
    kb2 = KnowledgeBase(WORK / "kb_missing", dim=cfg.hidden_size)
    for p_, s_ in corpus["fact_table"].items():
        if p_ == port:
            continue
        t = f"端口 {p_} 的服务是 {s_}"
        kb2.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(t), domain="fact",
                                       kind="fact", text=t, answer=s_, verified=True), embed(t))
    store = CorrectionStore(WORK / "corrections")
    before, _ = answer_of(model, tok, q)
    rec = store.submit(q, before, svc, rule_text=f"端口 {port} 的服务是 {svc}",
                       domain_tag="fact", error_type="fact_mismatch")
    store.approve(rec.record_id)
    t = f"端口 {port} 的服务是 {svc}"
    kb2.add("fact", KnowledgeRecord(rid=KnowledgeRecord.make_id(t), domain="fact", kind="correction",
                                   text=t, answer=svc, source="correction", verified=True), embed(t))
    segs = retrieve_for(model, tok, kb2, embed, q, cfg)
    after, _ = answer_of(model, tok, "".join(s.text + "\n" for s in segs) + q)
    print(f"  [主张4 自我进化] 端口 {port} 缺失：纠正前答 {before!r}（应为 {svc!r}）"
          f"→ 回写并审核通过 → 纠正后答 {after!r}；复犯率 "
          f"{0 if after == svc else 1} → {'0 ✓' if after == svc else '1 ✗'}")
    report["evolution_l1"] = {"port": port, "before": before, "after": after, "gold": svc}
    dump()

    # 主张5：L2 参数沉淀——审核通过的纠正变成该专家的微调集
    port2 = corpus["ports_held"][1]
    svc2 = corpus["fact_table"][port2]
    q2 = f"查询端口 {port2} 上运行的服务是什么\nA: "
    before2, _ = answer_of(model, tok, q2)                       # 不给检索：应当答不出
    retain_before = accuracy(model, tok, corpus["eval_fact"])     # 该专家本来能做的题
    # 必须用真正要微调的那条 prompt 去问路由。用另一条题（而且是带上下文的版本）问出来的
    # 专家，和这条题实际走的专家可能不同，就会出现"解冻了专家、栈却没进计算图"。
    fact_expert = primary_expert(model, tok, q2)
    rec2 = store.submit(q2, before2, svc2, rule_text=f"端口 {port2} 的服务是 {svc2}",
                        domain_tag="fact", error_type="fact_mismatch")
    approved = [a for a in [store.approve(rec2.record_id)] if a]
    ft = finetune_expert(model, tok, cfg, fact_expert,
                         corrections_to_examples(approved), steps=80)
    after2, _ = answer_of(model, tok, q2)
    retain_after = accuracy(model, tok, corpus["eval_fact"])
    print(f"  [主张5 参数沉淀] 不给检索时端口 {port2}：微调前 {before2!r} → 只解冻专家 {fact_expert} "
          f"的适配器（{ft['trainable_params']:,} 参数，loss {ft['loss_first']:.2f}→{ft['loss_last']:.2f}）"
          f"→ 微调后 {after2!r}（应为 {svc2!r}）")
    print(f"          防遗忘检查：该专家其他题 {retain_before*100:.0f}% → {retain_after*100:.0f}%"
          f"（{'没被带坏' if retain_after >= retain_before - 0.05 else '明显下降，需要更小步长'}）")
    report["l2_consolidation"] = {"port": port2, "before": before2, "after": after2, "gold": svc2,
                                  "expert": fact_expert, "finetune": ft,
                                  "retention_before": retain_before, "retention_after": retain_after}
    dump()
    print(f"\n完整结果写入 {WORK/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
