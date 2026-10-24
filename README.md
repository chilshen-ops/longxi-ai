# longxi-ai（KSR-MoE）

全新架构的大模型，知识库和专家解耦，由路由根据需求调用对应专家到显存计算，还由自我迭代功能。

完整需求（对齐后的版本）：知识库与专家解耦；路由按需求把对应专家调进显存计算；专家使用内存里的
知识库；显存常驻一个常识专家，用于修正和常识答复；并且由这个常驻常识专家在出错修正后负责维护
知识库。**要先对齐后再做。**

## 现在的主线：真模型聊天台（对话 + 工具 + 思考 + 外部知识库）

`tools/chatui.py` 把同一个主张搬到真的对话模型上（Qwen3-1.7B，fp16 跑在 P40 上，加载 20 秒）。
要验证的东西没变：**它记住的新事实存在模型外面，写进去不用重训，权重一个字节都不动。**

```powershell
# 必须用 .venv-gpu 的 python，而且不要设 PYTHONPATH（会 import 到 CPU 版 torch）
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
$env:KSR_CHAT_MODEL='A:\default-workspace\qwen3-1.7b'
.venv-gpu\Scripts\python.exe tools\chatui.py --port 8770   # 浏览器打开 http://127.0.0.1:8770
.venv-gpu\Scripts\python.exe tools\chatui.py --selftest    # 三条验证，不起服务
```

实测结果（跑出来的，不是设计意图）：

| | 0.6B | 1.7B |
|---|---|---|
| 对话 / 思考模式 | ✓ / ✓（推理会想歪） | ✓ / ✓ |
| 算术走工具 | ✓ 稳定 | ✓ 稳定 |
| 自己调 `kb_add` 写知识 | ✗ 不可靠：嘴上说"记下了"，工具没调 | ✓ 46 ms 写入，**权重指纹不变** |
| 自己去查知识 | ✗ 直接说"我不知道" | 不稳（有调有不调）→ 所以加了回答前自动检索 |
| 换个进程还记得 | — | ✓ 新进程没训练过任何东西，照样答出上一个进程写进文件的事实 |

模型自己决定去查这件事 1.7B 也不稳定，所以留了第二条路：回答前自动检索，把命中的写成
`【已知】` 注入上下文——页面会把这一步单独显示出来（`auto_retrieve`），能看到它查到了什么、
答案是怎么变的。工具路径也留着，两条各自测各自的数。

## 试过的路：给工具调用做 LoRA 微调（实测：整体不如底座）

流程三段：`tools/sftdata.py` 造 39600 条五类对话（查/写/不用工具/算/拒答；评测实体全部
不与 chatbench 重叠，逐个查过在训练数据里出现 0 次）→ `tools/finetune.py` 做 LoRA
（r=16 只注 qkvo，可训 6.42M 参数、占 0.37%；底座 fp16 冻结、可训练部分转 fp32）→
chatui 用 `KSR_CHAT_ADAPTER` 叠上 adapter（走 `merge_and_unload()`，推理路径和没 adapter
时逐字节一致，测出来的差别只是训练本身）→ `tools/chatbench.py` 两种模式各量一遍。

对照（同一台机器同一时段，两边都是新跑的；留出的是实体，句式在训练里出现过——测的是
学到的行为能不能泛化到没见过的事实，不是没见过的句子）：

| | 底座 | + adapter |
|---|---|---|
| 工具路径：该查时自己去查 | 4/6 | 3/6 ↓ |
| 工具路径：该写时自己去写 | 3/3 | 0/3 ↓ |
| 工具路径：不该动时动了 | 1/3 | **0/3**（修好了） |
| 自动检索：本身命中 | 6/6 | 6/6 |
| 自动检索：拿到之后认账 | 6/6 | 1/6 ↓ |
| 自动检索：拿到之前自己也去查 | 6/6 | 1/6 ↓ |

**只有误触发被修好了，其余全面变差。** 失败的样子和原因都定位到了：

- 过拟合是明显的：train loss 0.002、验证集 loss 0.094。合成数据句式有限，模型背下了模板。
- 没调工具的那些用例，模型输出的是**裸 JSON**（`{"name": "kb_add", ...}`，没有
  `​tool_call` 包装）——载荷学会了，包装丢了。
- auto 认账崩掉有结构性原因：`【已知】` 这种注入格式**训练数据里一次都没出现过**，
  微调把条件分布收窄到训练那一种，自动检索的提示词反而出了分布。
- 一个更要紧的事实：底座本来就会写（3/3）。这次微调瞄准的"该写 0/3"根本不存在——
  那个 0/3 是更早一个坏 adapter 的症状，不是底座的缺陷。

方法要在什么条件下才可能有用（还没做）：混入通用指令数据保泛化、训练里带上 `【已知】`
注入格式、句式多样性或按验证集早停。做之前该先问值不值——底座剩的两个毛病里，
误触发这条用"把什么不算知识写进工具描述"已经部分解决（2/3→1/3），这次微调把它修到 0/3，
是它目前唯一的净收益。

另外，"损失只算模型自己生成的那几轮"这件事有两个坑，**都是静默的**——不报错，
只是训出来的东西不能用：①先搜正文再回头验位置，会把用户消息里重复出现的同一句话
当成 assistant 正文（工具调用复述问题、改写任务原样返回都会踩到）；②框过
`<|im_end|>`，模型就去学"替用户说话"了。两个都被 `tests/test_finetune_encode.py`
钉死了，改 `encode()` 必须先过它。


## 先跑起来看看（30 秒）

演示台演示的是这个项目**唯一有硬证据**的能力：往知识库里加一条事实，不重训、不产生梯度，
模型立刻能答对。

**网页版**（推荐：浏览器打开就能用。只用 Python 标准库，不装任何包）：

```powershell
cd ksr-moe
$env:PYTHONPATH="$PWD\.venv-lib"; $env:PYTHONIOENCODING='utf-8'; $env:KSR_THREADS='2'
python tools\webui.py --work work-conflict --port 8765
# 浏览器打开 http://127.0.0.1:8765　（模型在后台加载约 40 秒，页面顶部会显示进度）
```

```bash
# Linux / macOS
PYTHONPATH=.venv-lib PYTHONIOENCODING=utf-8 python3 tools/webui.py --work work-conflict --port 8765
```

**命令行版**（同样的功能，没有终端就跳过它）：

```powershell
python tools\play.py --work work-conflict --demo    # 自动跑一遍：问 → 加一条 → 立刻答对
python tools\play.py --work work-conflict           # 交互模式
```

```bash
PYTHONPATH=.venv-lib PYTHONIOENCODING=utf-8 python3 tools/play.py --work work-conflict --demo
```

网页上是四个按钮：**问 / 加一条 / 改 / 删掉**。它刻意把两件事**并排**显示——"不检索时模型答
什么"和"带检索时模型答什么"——因为只看最终答案，就把"检索起了作用"和"模型本来就会"混成一回事。
检索到的片段如果不是在讲这个端口，页面会明说"没有这条知识"，不会把"照抄了一条不相干的条目"
算成"采纳了知识库"。

命令行里可以试这些：

| 想试什么 | 敲这个 | 会看到什么 |
|---|---|---|
| 加一条知识、不重训能不能答对 | `加 9100 svc00` | 写入 2.9 ms、**权重指纹不变**，同一个问题立刻答对 |
| 改一条知识（哪里不对修哪里） | `改 9005 svc03` | 先删旧条目再写新的，立刻生效 |
| 模型会不会不认账 | `问 9000` | 并排显示"不检索时答 X"和"带检索时答 Y"，并注明它采纳了谁 |
| 删一条知识 | `删 9101` | 从库里去掉，再问就回到"没有证据"的状态 |
| 这条能力的边界 | `加 9101 Panov` | 写入成功、检索也命中（混合分 1.93），但模型**不照抄**。查过原因：`Panov` 是它认识的词（词表里的一个整 token），不是读不出——它只在见过的答案形状（`svcNN`）里学会抄上下文，**认账是格式绑定的** |

**它现在是什么**：一个 220 万参数的原型，任务是合成的"端口→服务"，配一个可热更新的知识库。
**它不是什么**：不是聊天模型；也不演示"路由选专家"——实测把全部专家清零，答案一题都不变
（见 [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)）。演示台会把失败也如实打出来，不会只演成功的那次。

如果 `work-conflict/` 不在了，重新生成一次（约 10 分钟）：

```bash
KSR_WORK=work-conflict KSR_FRESH=1 KSR_CONFLICT=0.5 \
KSR_THREADS=2 OMP_NUM_THREADS=2 python -m ksr.experiment
```

## 怎么读这个项目

| 文件 | 内容 |
|---|---|
| [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) | **知识归属规则与基准**：一条知识该放哪里、怎么知道它真的在那里。每条规则都配一个能证伪它的测量 |
| [docs/REPORT.md](docs/REPORT.md) | 实测报告：测到了什么、没测到什么、为什么 |
| [docs/NUMBERS.md](docs/NUMBERS.md) | 全部规格数字，由 `tools/ksr_calc.py` 生成，**唯一数字来源** |
| [docs/results_scaled.json](docs/results_scaled.json) | 最近一次端到端实验的原始结果 |

## 怎么跑

```bash
# 依赖（CPU 版 torch + faiss，装到本地目录，不动系统环境）
pip install --target .venv-lib --index-url https://download.pytorch.org/whl/cpu torch faiss-cpu numpy
export PYTHONPATH="$PWD/.venv-lib"

python tools/ksr_calc.py --selfcheck          # 规格自校验，必须退出码 0
python tests/test_model_core.py               # 模型核心回归测试，当前 12/12
python tests/test_kb_retrieval.py             # 检索回归测试（不依赖库规模），当前 6/6
python -m ksr.experiment                      # 端到端：训练 + 六项主张实测
python tools/attrib.py --work work-scaled     # 知识归属探针：知识到底在专家/KB/哪里都没有
```

这台机器满载会硬断电，实验要限流：

```bash
KSR_THREADS=2 OMP_NUM_THREADS=2 python -m ksr.experiment
```

GPU 测量（`.venv-gpu` 是 cu126 独立环境）：

```bash
# 注意：用 GPU 的 python 时必须清掉 PYTHONPATH——它会优先于 site-packages，
# 于是 GPU 版 torch 会导入 CPU 版，cuda 直接"不可用"（这个坑踩过，症状是所有字段为空）
unset PYTHONPATH
python tools/bench.py --config tools/configs/ksr_180m.json --side ksr \
    --device cuda --dtype fp16 --batch 8 --hot-experts 4 --verify
```

`--verify` 会先证明新写的批量解码循环和 `model.generate()` 逐 token 一致，再谈吞吐数字。

微调这条线（都要 `.venv-gpu`、清掉 PYTHONPATH、`KSR_CHAT_MODEL` 指向模型目录；结论见上面
「试过的路」一节）：

```bash
python tools/sftdata.py                        # 造五类对话 39600+400 条，默认写 work-chat/sft/（实体不与评测集重叠）
python tools/finetune.py                       # LoRA，默认 cuda:1 / lr=1e-5（三档学习率实测见其文档），约 3.7h
KSR_CHAT_ADAPTER=work-chat/lora-v2/final python tools/chatbench.py   # 带 adapter 跑两种模式
python tests/test_finetune_encode.py        # 改 encode() 必须先过这个（无模型目录时自动跳过）
```

## 改完之后必须同步 GitHub（每一次都要）

这台机器上 git 装在 `C:\Program Files\Git\cmd\git.exe`，**不在 PATH 上**，要用完整路径：

```powershell
$g = 'C:\Program Files\Git\cmd\git.exe'
& $g -C A:\default-workspace\ksr-moe status --short
```

提交前必须全绿，三件都要：

```bash
python tests/test_model_core.py         # 12/12
python tests/test_kb_retrieval.py       # 6/6
python tools/ksr_calc.py --selfcheck    # 退出码 0
```

改过 `tools/finetune.py` 的 `encode()` 再加一件，用 `.venv-gpu` 跑（模型不在仓库里，
找不到 tokenizer 时它自己跳过并退出 0）：

```bash
python tests/test_finetune_encode.py    # 4/4
```

`.dl/`（下载缓存，2.4 GB）和 `work-*/`（实验产物）都在 `.gitignore` 里，不要提交。
`work-sup/` 是被跟踪的实验输出，改到它要先确认是有意的。

**推送需要 GitHub 凭据，而这台机器上一个都没有**（环境变量、`.git-credentials`、`.netrc`、
Windows 凭据管理器都查过）。两条路，任选其一：

```powershell
# 1) 界面授权（会弹浏览器登录）
$env:GCM_INTERACTIVE='always'; & $g push origin main

# 2) 用带 repo 作用域的 PAT（一次配好，之后不用再认证）
& $g remote set-url origin https://<用户名>:<PAT>@github.com/chilshen-ops/longxi-ai.git
& $g push origin main
```

这个仓库把 `core.autocrlf` 设成了 `input`：不改的话会有几个文件永远显示 modified，
而 `git diff` 里是空的（纯行尾差异），会把真实改动淹掉。同理，`git checkout -- <文件>`
可以还原这种没有内容差异的"假改动"。

## 结构

```
ksr/            模型与框架
  config.py       配置加载（未知键直接报错，不静默忽略）
  nets.py         混合注意力 + 每层路由 + 常驻常识专家 + 低秩专家栈（整栈原子加载）
                  专家分发按 argsort 分段，整层只做一次 host 同步；推理期可按 LRU 动态驻留
  knowledge.py    知识库（SSD+内存）、向量索引 + 符号倒排、三重检索硬预算、纠正存储
  selfevolve.py   L0/L1/L2 三档策略、L1 记忆级自我迭代、L2 参数级微调 + 防遗忘
  curator.py      常识专家当知识库策展人：新增/合并/否决 + 冲突标记（三条硬约束写死在代码里）
  tokenizer.py    贪心最长匹配 + 字节回退；训练与推理共用同一条编码路径
tools/ksr_calc.py 参数量 / FLOPs / 显存 / 冷专家搬运的解析计算器（含基线对比与自校验）
tools/bench.py    吞吐与显存实测：--device/--dtype/--batch，--side dense 生成同参数量稠密基线，
                  --hot-experts 测动态驻留（显存 vs 吞吐 vs 搬运次数）
tools/attrib.py   知识归属探针：gap_experts / gap_kb / 上界探针 / 重复权威，输出判定
tools/play.py     演示台（命令行）：加一条知识立刻能答对（--demo 自动演示，或直接进交互模式）
tools/webui.py    演示台（网页版）：只用标准库起服务，四个按钮 问/加/改/删，把"有没有检索"并排显示
tools/chatui.py   真模型聊天台：Qwen3-1.7B + 工具(calc/kb_add/kb_lookup) + 思考 + 外部知识库；
                  设 KSR_CHAT_ADAPTER 可叠 LoRA（merge_and_unload，推理路径与无 adapter 时一致）
tools/chatbench.py 分开量两条路：模型自己调工具（tool）vs 回答前自动检索（auto），含误触发
tools/sftdata.py   微调数据生成：查/写/不用工具/算/拒答五类，评测实体刻意不重叠
tools/finetune.py  LoRA 微调：底座 fp16 冻结、可训练参数转 fp32；损失只算 assistant 轮——
                  怎么框区间、踩过哪些静默的坑，见它 encode() 的文档和 tests/test_finetune_encode.py
tools/configs/    ksr7b_32L.json（7B 定稿）、ksr7b.json（紧凑版）、四个可运行原型
tests/            独立可跑的回归测试（不依赖 pytest）
```

## 当前结论摘要（详见 docs/KNOWLEDGE.md 的基准表）

**成立（有硬证据）**

- **知识外置**：留出端口上"检索开 92% / 检索关 4%"，检索命中率 100%，注入 28 token/题在预算内。
  答案确实不在权重里。（旧配方下是 75% / 8%，差距的原因见下一条。）
- **模型认账是训练出来的，不是检索出来的**：旧配方上限只有 75%，而把答案手抄进题面也只有 75%——
  检索和呈现都没有问题（命中 100%、格式 gap 0），是模型不肯照抄给它的证据（证据优先率 73%）。
  加 `KSR_CONFLICT=0.5` 做冲突改写训练后，同一评测升到 **92%**，证据优先率 **73% → 91%**，
  新事实从 59% 升到 **89%**，且仍等于手抄上界。副作用实测为 0（先验本来对、被检索带错 0 条）。
- **更新知识库不需要重训**：插入 200 条事实后权重指纹逐字节不变、单条 2.6 ms、跨重启仍生效，
  新事实准确率 **89% = 把答案手抄进题面的上界**。
- **自我纠错**：写错 → 回写 → 审核 → 答对，**复犯率 0**（这就是"不会反复踩同一个坑"）。
- **检索不随库规模退化**：2000 条库、候选池压到 32 时召回 286/286（修前 5/286）。

**不成立 / 未测出（原因已定位）**

- **效率**：同参数量稠密基线对照，GPU 上 KSR 是 0.70–0.99×，**batch 越大越差**。根因不是架构
  而是实现：分发开销随**候选专家数**增长（E=1→16 时吞吐 224→90，与每 token 计算量无关）。
  重写成分段分发后涨到 0.70–0.99×，但仍不到 1。混合注意力本身不吃亏（去掉 MoE 后与稠密持平）。
- **动态驻留**：省 28% 显存、赔 **12× 吞吐**。真实流量下容量 4 只罩住 54% 的选择，所以必然一直
  搬；按领域看是集中的（`linux→专家1 76%`）→ 单领域部署成立、混流不成立。设计要求是
  **按流量预置、只换入不换出**，不是请求期 LRU。
- **动态数量（自适应 top-k）**：没有可用信号——路由置信度 p25–p99 **全是 1.000**，阈值卡不掉任何
  token。这是 oracle 蒸馏把靶做得太可分导致的，要靠训练期信号解决，不是推理期阈值。
- **专家承载知识**：**没有**。把 16 个技能专家全部清零，三组题**没有任何一题的答案改变**
  （56%→56%、75%→75%、8%→8%）。把常识 FFN 也关掉、只剩 embedding + 注意力，仍有 43.8%
  （随机猜 10%）——模型确实会答的部分住在**必须常驻、路由动不了**的那部分里。
  把任务加难（每领域映射数 40、步数同步放大、答案集合固定不让词表膨胀）后重测，
  留出集放大到 128 题：开专家 39.1% / 清零 41.4%，**gap = −2.3pp ± 8.7pp**，与 0 无法区分。
  此前在 32 题留出集上测到的 +9pp、+6pp 都是噪声。
  第二个配置（`ksr_tiny.json`，4 层）上结论更强：常识 FFN 清零后只剩 **3.1%**、两个 FFN 都关是
  **0%**——知识全在必须常驻的权重里，而全部专家清零仍然是 **0 题变化**。
  结论：**"路由精准激活专家"在这台机器上不是没测好，是测不出来**——它要求两个条件同时成立
  （知识必须是算出来而非查出来的、专家要多到装不进显存），当前一个都不满足。详见
  [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)。
- **合成数据的工具调用微调，打不过底座**：39600 条五类对话做 LoRA（可训参数 0.37%），
  与基线同机同时段的对照是：该自己去查 4/6→3/6、该自己去写 3/3→0/3、auto 认账 6/6→1/6，
  只有误触发 1/3→0/3 是修好的。原因定位到三条：过拟合（train 0.002 / 验证 0.094）、
  坏的形式是输出裸 JSON（载荷学会了、`​tool_call` 包装丢了）、`【已知】` 注入格式在训练里
  从未出现（auto 路出了分布）。还有个前提性错误：瞄准的"该写 0/3"并不存在——底座本来 3/3，
  那个 0/3 是更早一个坏 adapter 的症状。细表见"试过的路"一节。
- **重复权威没有消除**：`work-conflict` 上仍有 **21 条**留出端口模型不给检索就自己给了一个
  存在的、错误的服务名（R4）。冲突改写训练让它"认账"了（证据优先率 91%），所以这个错误被压住，
  但两个权威来源并存这件事本身还在：一旦检索不到，模型就会自信地答错。
  要做的是让常识专家在**没有证据时拒答**，而不是猜——这条还没有实现，也没有对应指标。

**这个尺度测不出**：7B。6.84B 总参 / 3.77B 每 token 激活 / 11.9 GFLOPs decode = 稠密 7B 的 38% /
常驻 14.29GB 都只有解析数字。这台机器物理内存 16 GB，6.84B 参数连 CPU 侧初始化都放不下
（fp32 要 27 GB），所以 7B 端到端测不了。
