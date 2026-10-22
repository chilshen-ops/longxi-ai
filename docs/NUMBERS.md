{
  "config": "tools\\configs\\ksr7b.json",
  "result": {
    "name": "KSR-MoE 7B 定稿（低秩旁路 r=512 档）",
    "cfg": {
      "name": "KSR-MoE 7B 定稿（低秩旁路 r=512 档）",
      "vocab_size": 131072,
      "hidden_size": 4096,
      "num_layers": 24,
      "num_attention_heads": 32,
      "num_kv_groups": 4,
      "head_dim": 128,
      "full_attn_layers": [
        1,
        5,
        9,
        13,
        17,
        21
      ],
      "linear_attn": {
        "num_kv_groups": 4,
        "num_v_heads": 32,
        "head_dim_k": 128,
        "head_dim_v": 128
      },
      "ffn_type": "swiglu",
      "reasoner_hidden": 1024,
      "num_skill_experts": 16,
      "expert_rank": 512,
      "expert_ffn_type": "swiglu",
      "top_k_experts": 1,
      "router_hidden": 512,
      "corrector": {
        "enabled": true,
        "placement": "sequence",
        "hidden": 4096,
        "num_classes": 3,
        "with_correction_proj": true,
        "always_on": false
      },
      "tied_embeddings": true,
      "context_len": 32768,
      "quant": {
        "attention": "bf16",
        "reasoner": "bf16",
        "router": "int8g64",
        "experts": "int4g32",
        "embedding": "bf16",
        "kv_cache": "bf16",
        "corrector": "bf16"
      },
      "hot_experts": 16,
      "link": {
        "bandwidth_gbps": 56.0
      },
      "runtime_overhead_bytes": 1000000000,
      "limits": {
        "max_total_params": 7500000000,
        "max_activated_params": 2200000000,
        "max_resident_gib": 12,
        "max_expert_pool_params": 5000000000
      },
      "_path": "tools\\configs\\ksr7b.json"
    },
    "params": {
      "attention_full": 226492416,
      "attention_linear": 679477248,
      "attention": 905969664,
      "reasoner": 301989888,
      "experts_pool": 2415919104,
      "experts_unit": 6291456,
      "router": 50552832,
      "corrector": 50343936,
      "embedding": 536870912,
      "lm_head": 0,
      "norms": 204800,
      "total": 4261851136,
      "activated_per_token": 1946583040
    },
    "flops": {
      "proj_per_token": 3893166080,
      "prefill_total": 181584774823936,
      "decode_per_token": 7152140288,
      "attn_extra_prefill": 52776558133248,
      "attn_extra_dense": 211106232532992,
      "attn_ratio_vs_dense_same_depth": 0.25
    },
    "vram": {
      "attention": 1811939328.0,
      "reasoner": 603979776.0,
      "router": 51342720.0,
      "experts_hot": 1358954496.0,
      "experts_cold": 0.0,
      "experts_all": 1358954496.0,
      "embedding": 1073741824.0,
      "corrector": 100687872.0,
      "kv_cache": 402653184.0,
      "linear_states": 18874368.0,
      "overhead": 1000000000,
      "weights_total": 5000646016.0,
      "resident": 6422173568.0
    },
    "offload": {
      "hot_experts": 16,
      "cold_experts": 0,
      "cold_stack_bytes": 84934656.0,
      "load_ms": 1.5166902857142859,
      "bandwidth_gbps": 56.0
    },
    "layers": {
      "total": 24,
      "full_attn": 6,
      "linear_attn": 18
    },
    "context_len": 32768
  },
  "limit_failures": []
}