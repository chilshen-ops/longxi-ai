r"""LoRA 微调：教模型「什么时候该调哪个工具、什么时候别调」。

为什么用 LoRA 而不是全参——都是实测出来的，不是猜的：
  · 1.7B 全参 fp16：峰值 16.4 GB、524 tok/s；只训 13.7% 参数：6.2 GB、595 tok/s。
  · 主机内存只剩 4.9 GB，而 3B 的 fp32 优化器状态要 36 GB，连 checkpoint 都放不下。
  · P40 没有 bf16，标准 AMP（主权重 fp32）在 1.7B 上直接 OOM。
所以基座权重保持 fp16 冻结，只训 LoRA 旁路，并把可训练参数转成 fp32——
这样梯度是 fp32，不会在 fp16 下下溢（LoRA 的 B 初始化为 0，fp16 梯度很容易全变成 0）。

损失只算 assistant 那几轮（工具调用和最终回答）。system、用户的话、工具返回的结果
都是"别人说的话"，模型没生成过，算进损失会让它去拟合不该拟合的东西。

用法：
    .venv-gpu\\Scripts\\python.exe tools/finetune.py --device cuda:1

实测过的三档学习率（都是全量 39600 条，约 3.7 小时跑完）：
  · 2e-4：step 1300 之后 loss 变 NaN，彻底发散。
  · 1e-4：不 NaN，loss 也降，但生成的工具调用是坏的（logits 爆炸，
    生成到工具调用那段就报 CUDA assert）。
  · 1e-5：曲线平（train loss ~0.002），但这已经是过拟合的水平
    （验证集 loss 0.094），chatbench 实测仍整体不如底座——见 README。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
# 只为了拿推理那份 TOOLS，不加载模型。训练和推理必须渲染出同一个提示词，
# 这一点下面 encode() 里会用到，见那里的注释。
import chatui as C  # noqa: E402

IGNORE = -100

# 模板给每一轮的结束符。正文后面紧跟的就是它，框到它为止。
END_MARK = "<|im_end|>"
# 一轮 assistant 的开头。模板会给最后一轮 assistant 插一个空的 think 块，
# 所以正文不一定紧跟在这个开头后面（见 encode 里怎么跳过 think 块）。
ASSISTANT_OPEN = "<|im_start|>assistant\n"


def encode(tok, msgs, tools=None):
    """整段渲染，把每个 assistant 轮的正文框成损失区间；区间外一律 IGNORE。

    先说不用什么办法。原来用的是「分别渲染前缀、拿 token 数相减」，实测是错的：
    Qwen3 的模板会在最后一个 assistant 轮前插入空的 think 块，而带
    add_generation_prompt 的前缀渲染不插，两边一相减区间就整体错位——
    诊断脚本里看到框出来的内容把下一轮「工具返回」的开头也算进去了。
    逐条消息套模板再拼起来也不行：模板只在第一条前加 BOS，边界会错。

    改成的办法是**先锚到这一轮 assistant 的开头**（<|im_start|>assistant\n，
    跳过模板可能插的空 think 块），确认正文紧跟其后，把「正文 + 结束符」
    框成损失区间，其余一律 IGNORE。前一段字符串是整段的前缀，token 序列
    必然也是前缀关系，所以「数前缀有多少 token」就是正文的 token 起点。

    两个坑，都是实测踩出来的（钉在 tests/test_finetune_encode.py）：
      * **不能反过来找**（先搜正文、再验它前面是不是 assistant 开头）：
        正文里的字很可能在别处一模一样地出现过——最常见的是 assistant
        复述用户的话（工具调用带 query、改写任务原样返回那句话）。
        反过来找会把用户消息里那份当成正文，验不过就整条丢掉，
        丢掉的正好是最该学的那批数据（实测一条都框不出来）。
      * **框到 <|im_end|> 就停**：再往后是下一轮的用户消息和工具返回，
        那些是「别人说的话」，模型没生成过，框进去模型就去学"替用户
        说话"了。这个错误不会报错，只是训出来的东西不能用。
    所以框完必须回头核对（见下面的两处），对不上就整条丢掉：
    宁可少训一条，不要学错一段。

    tools 必须和推理时传的一样（chatui 那边是 tools=TOOLS）：模板会把工具的 JSON schema
    拼进 system 段。训练时不传，模型见到的上下文就比推理时少一截，
    学出来的行为挂在「没有 schema 的提示词」上，到推理时条件对不上。
    """
    kw = {"tools": tools} if tools else {}
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [IGNORE] * len(ids)
    # 做法和为什么，都写在上面 docstring 里；这里只标每一步在防什么。
    search_from = 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        content = m["content"]
        mh = full.find(ASSISTANT_OPEN, search_from)
        if mh < 0:
            return None, None
        pos = mh + len(ASSISTANT_OPEN)
        if full.startswith("<think>", pos):
            t = full.find("</think>", pos)
            if t < 0:
                return None, None
            pos = t + len("</think>")
            while pos < len(full) and full[pos] == "\n":
                pos += 1
        if not full.startswith(content, pos):
            return None, None
        cs = pos
        m_end = full.find(END_MARK, cs + len(content))
        if m_end < 0:
            return None, None
        if full[cs + len(content):m_end].strip():
            return None, None              # 正文和结束符之间夹了东西，位置不可信
        ce = m_end + len(END_MARK)
        start = len(tok(full[:cs], add_special_tokens=False)["input_ids"])
        end_tok = len(tok(full[:ce], add_special_tokens=False)["input_ids"])
        content_end = len(tok(full[:cs + len(content)], add_special_tokens=False)["input_ids"])
        if tok.decode(ids[start:content_end]) != content:
            return None, None
        labels[start:end_tok] = ids[start:end_tok]
        search_from = cs + len(content)
    return ids, labels


def load_rows(path, tok, max_len, limit=0, tools=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids, labels = encode(tok, json.loads(line), tools)
        if ids is None:      # 位置对不上或解码核对不过（见 encode），整条丢
            continue
        if sum(1 for x in labels if x != IGNORE) < 4:      # 没有可学的目标，训了只会添噪声
            continue
        rows.append((ids[:max_len], labels[:max_len]))
        if limit and len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def val_loss(model, rows, device, n=60):
    model.eval()
    tot, cnt = 0.0, 0
    for ids, labels in rows[:n]:
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=y).loss
        if torch.isfinite(loss):
            tot += float(loss)
            cnt += 1
    model.train()
    return tot / max(cnt, 1)


def main() -> int:
    import os
    ap = argparse.ArgumentParser(description="LoRA 微调（只训低秩旁路）")
    ap.add_argument("--model", default=os.environ.get("KSR_CHAT_MODEL",
                                                     r"A:\default-workspace\qwen3-1.7b"))
    ap.add_argument("--data", default=str(ROOT / "work-chat" / "sft" / "train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "work-chat" / "sft" / "val.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "work-chat" / "lora"))
    ap.add_argument("--device", default="cuda:1",
                    help="默认用第二张卡：聊天台占着 cuda:0，两边互不影响")
    ap.add_argument("--max-len", dest="max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="默认 1e-5：2e-4 实测发散（step 1300 后 loss 变 NaN），"
                         "1e-4 不 NaN 但工具调用生成坏掉，只有 1e-5 稳定")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-every", dest="save_every", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="只训前 N 条，用来先试通流程")
    ap.add_argument("--no-tools", dest="tools", action="store_false", default=None,
                    help="不给模板传工具 schema（默认传，和推理一致）")
    args = ap.parse_args()
    tools = None if args.tools is False else C.TOOLS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data, tok, args.max_len, args.limit)
    vrows = load_rows(args.val, tok, args.max_len, 60)
    toks = sum(len(a) for a, _ in rows)
    print(f"训练 {len(rows)} 条 / {toks:,} token；验证 {len(vrows)} 条；设备 {dev}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                device_map=dev, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    # 基座留 fp16（省显存），可训练部分转 fp32：P40 没有 bf16，LoRA 的 B 初始为 0，
    # 梯度要是也存成 fp16 很容易整体下溢成 0，训了等于没训
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数 {sum(p.numel() for p in trainable):,}"
          f"（{sum(p.numel() for p in trainable) / model.num_parameters() * 100:.2f}%）",
          flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    order = list(range(len(rows)))
    import random
    random.Random(0).shuffle(order)
    step, seen, run, run_tok, t0 = 0, 0, 0.0, 0, time.time()
    for idx in order:
        ids, labels = rows[idx]
        x = torch.tensor([ids], device=dev)
        y = torch.tensor([labels], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=y).loss / args.accum
        if not torch.isfinite(loss):
            continue
        loss.backward()
        run += float(loss.detach()) * args.accum
        run_tok += len(ids)
        seen += 1
        if seen % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                rate = run_tok / el
                eta = (len(rows) - (step * args.accum)) * (len(ids) / max(rate, 1)) / 3600
                print(f"step {step:>5} loss {run / (50 * args.accum):.4f} "
                      f"已用 {el / 3600:.2f}h  {rate:,.0f} tok/s  预计还需 {eta:.1f}h",
                      flush=True)
                run, run_tok, t0 = 0.0, 0, time.time()
            if step % args.save_every == 0:
                v = val_loss(model, vrows, dev)
                print(f"  step {step} 验证损失 {v:.4f}", flush=True)
                model.save_pretrained(out / f"step-{step}")
    v = val_loss(model, vrows, dev)
    print(f"训练结束 验证损失 {v:.4f}", flush=True)
    model.save_pretrained(out / "final")
    print(f"适配器已存 {out / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
