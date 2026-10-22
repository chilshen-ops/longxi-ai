# longxi-ai（KSR-MoE）

全新架构的大模型，知识库和专家解耦，由路由根据需求调用对应专家到显存计算，还由自我迭代功能。

完整需求（对齐后的版本）：知识库与专家解耦；路由按需求把对应专家调进显存计算；专家使用内存里的
知识库；显存常驻一个常识专家，用于修正和常识答复；并且由这个常驻常识专家在出错修正后负责维护
知识库。**要先对齐后再做。**

## 怎么读这个项目

| 文件 | 内容 |
|---|---|
| [docs/REPORT.md](docs/REPORT.md) | **实测报告**：测到了什么、没测到什么、为什么。先读这个 |
| [docs/NUMBERS.md](docs/NUMBERS.md) | 全部规格数字，由 `tools/ksr_calc.py` 生成，**唯一数字来源** |
| [docs/results_scaled.json](docs/results_scaled.json) | 最近一次端到端实验的原始结果 |

## 怎么跑

```bash
# 依赖（CPU 版 torch + faiss，装到本地目录，不动系统环境）
pip install --target .venv-lib --index-url https://download.pytorch.org/whl/cpu torch faiss-cpu numpy
export PYTHONPATH="$PWD/.venv-lib"

python tools/ksr_calc.py --selfcheck          # 规格自校验，必须退出码 0
python tests/test_model_core.py               # 模型核心回归测试，当前 10/10
python -m ksr.experiment                      # 端到端：训练 + 五项主张实测
```

实验默认单线程不安全，多核机器上建议限流（这台机器满载会硬断电）：

```bash
KSR_THREADS=2 OMP_NUM_THREADS=2 python -m ksr.experiment
```

## 结构

```
ksr/            模型与框架
  config.py       配置加载（未知键直接报错，不静默忽略）
  nets.py         混合注意力 + 每层路由 + 常驻常识专家 + 低秩专家栈（整栈原子加载）
  knowledge.py    知识库（SSD+内存）、向量索引、三重检索硬预算、纠正存储
  selfevolve.py   L0/L1/L2 三档策略、L1 记忆级自我迭代、L2 参数级微调 + 防遗忘
  curator.py      常识专家当知识库策展人：新增/合并/否决 + 冲突标记（三条硬约束写死在代码里）
  tokenizer.py    贪心最长匹配 + 字节回退；训练与推理共用同一条编码路径
tools/ksr_calc.py 参数量 / FLOPs / 显存 / 冷专家搬运的解析计算器（含基线对比与自校验）
tools/configs/    ksr7b_32L.json（7B 定稿）、ksr7b.json（紧凑版）、三个可运行原型
tests/            独立可跑的回归测试（不依赖 pytest）
```

## 当前结论摘要（详见 docs/REPORT.md）

**成立**：知识外置。留出端口上"检索开 67% / 检索关 4%"，三次独立运行复现（46%→54%→67%），
对照组始终 0-4%，检索命中率 100%，注入 28 token/题在预算内。答案确实不在权重里。

**机制通了但效果不稳**：L2 参数级自我迭代——只解冻一个专家的 196,608 个参数，
loss 8.95→2.25，防遗忘检查 67%→67%（没被带坏，这是"解耦"唯一被直接测到的好处），
但答案没改对。

**未测出，原因已定位**：路由分工。路由置信度 0.062 ≈ 1/16（均匀），而专家输出要乘上路由权重，
所以强制任何专家都不改变答案。根因是交叉熵收敛后负载均衡辅助损失反过来主导梯度，把路由按回均匀
——需要给路由加监督（语料本来就有领域标签），并把三档阈值改成分位数/温度校准。

**这个尺度测不出**：效率。2-4M 参数时调用开销远大于 FLOP 节省（实测 0.21×），
所以效率主张只能用解析数字：7B 定稿 6.84B 总参 / 3.77B 每 token 激活 /
11.9 GFLOPs decode = 稠密 7B 的 38% / 常驻 14.29GB。
