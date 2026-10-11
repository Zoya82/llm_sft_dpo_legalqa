# experiments/ — 支撑项目结论的实验脚本

这些脚本产生了写进 `legal_check.py` 注释和 `PROJECT_REPORT.md` 的关键结论。
放在这里是为了**让结论可复现**——文档里每一个"实测发现",都能在这里找到依据。

所有脚本直接 `python3 experiments/xxx.py` 运行(路径已写死为项目根)。

| 脚本 | 回答什么问题 | 结论落在哪 |
|---|---|---|
| `scan_n.py` | 字面比对的 n-gram 该取几?扫描 n=2~6 | `legal_check._char_overlap` 的注释:n=2 负例噪声高、n≥4 误杀转述,故取 3 |
| `calib_rerank.py` | bge-reranker 能不能做语义复核?标定阈值 | **否决依据**:正例区间 [7.81,10.31] 与否定反转 [7.77,10.31] 完全重叠,对否定反转识别率 0% |
| `calib_nli.py` | NLI 在法条核验上的真实表现 | `legal_check.semantic_recheck` 的注释:转述 99%、否定反转 72%,加否定词密度后漏网降到 12% |
| `retrieval_ab.py` | RAG 检索命中率低,是召回还是排序的问题?HyDE 和提高 k 有用吗 | PROJECT_REPORT §4.6:召回上限 73%,HyDE 单用无效(0.5B 当不了教师),问题+HyDE k=8 最高 48% |
| `rag_v2.py` | RAG 的 prompt 从封闭式改开放式、k 从 3 提到 5,有用吗 | PROJECT_REPORT §4.6:字数 109→128、有引用率纹丝不动 → **排除 prompt 假设** |
| `make_blind.py` | 生成 pairwise 盲评数据(分层抽样 + 随机打乱 A/B + 隐去模型名) | PROJECT_REPORT §4.5 / §4.6 的三组盲评 |
| `forgetting_probe.py` | 灾难性遗忘评测:领域外探针 + LoRA 缩放扫描 | PROJECT_REPORT §4.7 |

## 为什么这些脚本值得留

项目里有三次"我推翻了自己的结论",每一次的依据都在这些脚本里:

1. **核验器报"训练数据 18% 引用有问题"** → 抽查四条发现是核验器的 bug(见 `legal_check.py` 的"之一"索引注释);
2. **reranker 的"安全阈值 7.0"** → `calib_rerank.py` 用合成的轻度转述标定,而真实深度转述只有 4.92 分,阈值失效;
3. **RAG 回答塌缩** → 先怪检索(`retrieval_ab.py` 排除)、再怪 prompt(`rag_v2.py` 排除),最后定位到模型 in-context 能力。

**没有这些脚本,文档里的数字就只是断言。**

## 后训练数据改造阶段新增

| 脚本 | 回答什么问题 | 结论落在哪 |
|---|---|---|
| `reject_sampling.py` | DPO 负样本从 off-policy 换成 on-policy,能造出多少、质量如何 | PROJECT_REPORT §4.8:522/600 条(87%),96% 含引用错误;**模型真实错误是 misattributed 540 条 vs not_found 2 条**,与规则劣化的分布几乎不重叠 |
| `probe_negative_difficulty.py` | 三组偏好数据(纯规则/混合/纯难样本)各跑 40 步,看 accuracies 起点 | 未跑完(会话中断),留作 04 结果异常时的诊断工具 |
| `score_sft_quality.py` | SFT 训练数据的质量分布,能否用规则筛出 top-N | PROJECT_REPORT §4.10:**规则打分器与人工评分 r=0.258**,且无单调性 → 不可用于精排 |
| `sample_for_labeling.py` | 生成分层校准集,供强模型逐条打分 | 产出 `outputs/eval/labels/`,是上面那个 r=0.258 的来源 |
| `rag_v3_cite.py` | 在 prompt 里硬性要求逐点标注 + 给格式示例,能否解决"用了检索内容不写出处" | PROJECT_REPORT §4.10:**三版 prompt 有引用率都是 22%**,把"指令遵循弥补不了缺失的行为模式"从推断变成实证 |

## 这一阶段又推翻了三个判断

4. **「裸标题合并占 5%」** → 写检测器全池严格统计,实测 **0.2%**。5% 是拿 38 条里的 2 条外推来的,样本量根本不支撑;
5. **「规则打分器能筛数据」** → 人工评 38 条算相关性,**r=0.258**,而且我评 4 分的样本规则分中位比 5 分的还高,完全没有单调性;
6. **「文档里的 51% 算错了」** → 错的是我。3-gram 重合要问的是"回答里有多少内容来自条文",我算成了反方向 —— 回答仅 101 字、条文动辄几百字,方向反了会被长度稀释(28% vs 44%)。

**全项目共 14 条结论被自己的数据推翻(清单见 PROJECT_REPORT §5.5),多数根因是同一个:新造的指标本身没被验证就拿来下结论。**
| `precision_ab.py` | MPS 上 bf16 能不能替代 fp32 做训练 | PROJECT_REPORT §4.9 附:**bf16 快 292%、省 46% 显存,但 DPO 的 loss/accuracies 系统性变差**(10 个点 9 个同向)→ DPO 用 fp32,SFT 可试 bf16 |
| `watch_then_sleep.sh` | 守着长任务跑完后自动休眠(`caffeinate -w` + `pmset sleepnow`) | 工具脚本 |
