"""模型核心正确性测试。

不依赖 pytest：直接 `python tests/test_model_core.py` 也能跑。
每条测试都对应原稿代码里的一个真实缺陷，或一条规格与代码的一致性约束。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from ksr.config import KSRConfig
from ksr.nets import CacheList, ExpertStore, KSRModel
import ksr_calc

CFG_PATH = ROOT / "tools" / "configs" / "ksr_tiny.json"


def _model(seed: int = 0):
    cfg = KSRConfig.from_json(CFG_PATH)
    torch.manual_seed(seed)
    return KSRModel(cfg).eval(), cfg


def test_shapes():
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (2, 17))
    logits, meta = model(ids)
    assert logits.shape == (2, 17, cfg.vocab_size), logits.shape
    assert len(meta["confidence"]) == cfg.num_layers
    assert meta["confidence"][0].shape == (2, 17)


def test_prefix_invariance():
    """原稿缺陷 #2：prefill 没有因果掩码 → 后面的 token 会污染前面的预测。"""
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(ids)
        prefix, _ = model(ids[:, :7])
    assert torch.allclose(full[:, :7], prefix, atol=1e-4), (full[:, :7] - prefix).abs().max()


def test_router_gets_gradient():
    """原稿缺陷 #1：top-1 之后再 softmax 恒等于 1 → 门控零梯度，路由永远学不动。"""
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (2, 16))
    logits, meta = model(ids)
    loss = logits.float().sum() + cfg.aux_loss_weight * meta["aux_loss"]
    loss.backward()
    grads = [b.router.expert_head.weight.grad for b in model.blocks]
    norms = [float(g.norm()) for g in grads]
    assert all(g is not None for g in grads), "路由没有梯度"
    assert min(norms) > 1e-6, f"路由梯度范数过小: {norms}"


def test_confidence_is_not_constant():
    """原稿的 confidence 恒等于 1，阈值完全失效。这里必须是可区分的连续值。"""
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        _, meta = model(ids)
    conf = torch.stack(meta["confidence"])
    assert conf.std() > 1e-4, f"置信度几乎是常数: std={conf.std()}"
    assert (conf > 0).all() and (conf <= 1.0 + 1e-6).all()


def test_activated_params_match_calculator():
    """规格与代码必须逐项一致：模型自测的激活参数 == 计算器给出的数字。"""
    model, cfg = _model()
    raw = json.loads(CFG_PATH.read_text("utf-8"))
    want = ksr_calc.analyze(raw)["params"]["activated_per_token"]
    got = model.activated_params()
    assert got == want, f"模型 {got:,} vs 计算器 {want:,}（差 {got - want:,}）"


def test_total_params_within_model():
    model, cfg = _model()
    total = sum(p.numel() for p in model.parameters())
    raw = json.loads(CFG_PATH.read_text("utf-8"))
    want = ksr_calc.analyze(raw)["params"]["total"]
    assert total == want, f"模型 {total:,} vs 计算器 {want:,}（差 {total - want:,}）"


def test_kv_cache_matches_non_incremental():
    """增量解码（KV Cache）必须与整段前向逐 token 一致。"""
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (1, 9))
    with torch.no_grad():
        full, _ = model(ids)
        cache = CacheList(cfg.num_layers)
        outs = []
        for t in range(9):
            lg, _ = model(ids[:, t:t + 1], cache, t)
            outs.append(lg[:, -1])
    inc = torch.stack(outs, dim=1)
    # 增量解码第 t 步得到的就是位置 t 的 logits，与整段前向逐位置对应（不平移）
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_reasoner_always_active_and_experts_are_stacks():
    """原稿缺陷 #3：热专家缓存把 layer0 的权重当成全局权重 → 所有层共用 layer0。"""
    model, cfg = _model()
    store: ExpertStore = model.experts
    stack_params = [p for p in store.pool[1].parameters()]
    assert len(stack_params) == 3 * cfg.num_layers, "一个专家必须是每层一片的整栈"
    # 不同层的同专家权重必须不同（否则就是原稿那个 bug）
    a = store.pool[0][0].gate.weight.detach()
    b = store.pool[0][1].gate.weight.detach()
    assert not torch.allclose(a, b)


def test_expert_hot_set_respects_capacity():
    model, cfg = _model()
    store = model.experts
    store.cfg.hot_experts = 2
    store.set_hot([0, 1])
    assert store.hot == [0, 1]
    assert store.ensure_hot(2) is True          # 触发搬运
    assert store.hot == [1, 2] and store.evict_events == 1
    assert store.ensure_hot(1) is False         # 已在热集合，不搬运
    assert store.load_events == 1


def test_generate_runs_with_forced_expert():
    model, cfg = _model()
    ids = torch.randint(4, cfg.vocab_size, (6,)).tolist()
    out, agg = model.generate(ids, max_new=5)
    assert len(out) == 5
    assert set(agg) == {"confidence", "need_knowledge", "primary_expert"}
    assert 0.0 <= agg["confidence"] <= 1.0
    forced, agg2 = model.generate(ids, max_new=3, force_expert=2)
    assert len(forced) == 3


def test_paging_capacity_is_enforced_during_forward():
    """动态驻留的容量必须真的封顶——前向不能把热集合撑回"全部专家"。

    这条守的是一个真实犯过的错：enable_paging 只设了初始热集合、没设 cfg.hot_experts，
    于是 ensure_hot 的换出判据还是"装得下就加"，跑一轮下来 hot 又长回全部专家，
    峰值显存一点没降、搬运全落在预热阶段（稳态计数为 0）——看起来像"动态驻留没用"，
    其实是容量根本没生效。
    """
    model, cfg = _model()
    store = model.experts
    store.enable_paging(torch.device("cpu"), hot=[0], capacity=1)
    assert store.cfg.hot_experts == 1, "容量没有写进 cfg，换出判据还是旧值"
    assert store.hot == [0]
    ids = torch.randint(4, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        model(ids)                                       # 前向会按需 ensure_hot
    assert len(store.hot) <= 1, f"热集合被前向撑回 {len(store.hot)} 个：{store.hot}"
    assert store.resident_bytes <= store.stack_bytes * 1 * 4 + 1, "常驻字节统计没跟着封顶"


def test_stack_bytes_is_one_full_stack_not_layers_again():
    """stack_bytes = 一个专家整栈的参数量。pool[0] 本身就含每层一片，不能再乘层数。"""
    model, cfg = _model()
    store = model.experts
    per_stack = sum(p.numel() for p in store.pool[0].parameters())
    assert store.stack_bytes == per_stack, (
        f"stack_bytes={store.stack_bytes} 而一个栈是 {per_stack}，层数被乘了两遍")


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
