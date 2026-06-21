# 垂直领域小 LLM 的后训练对齐：SFT + DPO（中文法律问答）

> 这是 T5 问答项目的**升级版**。把"会微调一个模型"升级成"在垂直域对 decoder-only **大模型**做 **SFT + 偏好对齐(DPO)** 并建立评测体系"。
> 目标岗位：大模型**应用算法 / 后训练**方向。

## 数据集（真实，非自造）
- **DISC-Law-SFT**（复旦大学 FudanDISC 开源）法律问答子集，约 9.3 万条真实中文法律咨询问答。
- 来源：https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT
- `notebooks/01` 会自动下载一份样本（默认 2000 条，走国内镜像 hf-mirror）；下不动时回退到内置 `data/seed_clauses.jsonl`。
- ⚠️ 数据集版权归原作者，**请勿把下载的数据再传到你的公开仓库**（已在 `.gitignore` 忽略）；使用/引用请遵循其 License 并注明出处。

---

## 一、它补上了 T5 项目的三个缺口

| 缺口 | T5 项目 | 本项目 |
|---|---|---|
| ① 模型代际 | mengzi-T5（2020，encoder-decoder） | **Qwen2.5（decoder-only LLM）** + **LoRA/PEFT** |
| ② 偏好对齐 | 只有 SFT | **加了 DPO**（偏好优化） |
| ③ 叙事归属 | 课程作业、问题生成任务 | 独立项目、用**真实法律数据集**、贴简历领域 |

> 评测沿用了 T5 项目里你已经会的方法论：**消融对比表 + baseline + BLEU**，并升级加入 **LLM-as-judge**。

---

## 二、清单：要做完的 6 件事（= 6 个面试谈资）

- [ ] **01 任务与数据探查**：定义任务、看清数据格式、想清楚输入输出。
- [ ] **02 构造 SFT + DPO 数据**：把种子条款扩成指令数据；用"劣化规则"造 DPO 偏好对（chosen/rejected）。→ *命中 JD：数据合成/偏好数据*
- [ ] **03 SFT 微调（Qwen2.5 + LoRA）**：跑通 LoRA 微调，画 loss，看训练前后对比。→ *命中 JD：SFT、PEFT*
- [ ] **04 DPO 偏好对齐**：在 SFT 模型上做 DPO。→ *命中 JD：DPO、偏好优化*
- [ ] **05 评测与消融对比**：base / +SFT / +SFT+DPO 三档对比，自动指标 + LLM-as-judge + badcase。→ *命中 JD：模型评测、生产级*
- [ ] **写 README + 整理 GitHub**：用**你自己的话**重写本文件的"结论"部分。

---

## 三、运行顺序与环境

```bash
# 1. 装依赖（已有 torch/transformers/accelerate，只补这几个）
python3 -m pip install -r requirements.txt

# 2. 按顺序跑 notebooks/ 下的 01 → 05
#    每个 notebook 顶部都有一个【配置区】：run_mode = "smoke"(默认,快) / "full"(完整)
```

- 默认模型 `Qwen2.5-0.5B-Instruct`（小、MPS 上快，先跑通）；想要更好效果改成 `Qwen2.5-1.5B-Instruct`。
- 首次会从 HuggingFace 下载模型（约 1GB），需联网。下不动可设镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- Mac 用 **MPS**；代码默认 float32（MPS 上最稳）。

---

## 四、产物（跑完后会生成）

```
outputs/
  sft_merged/        # SFT 后合并的完整模型
  dpo_merged/        # SFT+DPO 后合并的完整模型
  figures/           # loss 曲线、消融对比图
  eval/              # 三档模型的预测、指标、badcase
  ablation_table.md  # 最终消融对比表（简历就引用这张表的数字）
```

---

## 五、结论

**做了什么**：在复旦 DISC-Law 中文法律问答数据上，对 Qwen2.5 完成 SFT + DPO 后训练全流程——数据清洗去重与偏好数据构造（5 类劣化规则）→ LoRA 微调 → DPO 偏好对齐 → 三档消融评测；并针对评测暴露的法条幻觉，补充 RAG 检索增强 baseline。

**消融结果**（base / +SFT / +SFT+DPO，评测集与训练隔离）：

| 模型 | 字符 BLEU | 规则 judge |
|---|---|---|
| base | 0.051 | 0.787 |
| +SFT | 0.058 | 0.779 |
| +SFT+DPO | **0.065** | 0.752 |

- **SFT**：train loss 2.3 → 1.6，BLEU 0.051 → 0.058，模型学会专业法律口吻与"引用法条"的输出格式。
- **DPO**：BLEU 进一步升到 0.065，rewards / accuracies 收敛至约 1.0。

**关键发现（比分数更重要）**：
- judge 分数逐档**下降**，深挖发现是规则 judge 在奖励"关键词堆砌"——说明**评测指标设计往往比训练本身更关键**，单一指标不可靠。
- DPO accuracies 轻松收敛到约 1.0，是因为偏好对用"劣化规则"构造、过于易区分；真实场景需要更"难"的 rejected（如用强模型生成"看似合理但有事实错误"的答案）。
- badcase 显示**法条事实幻觉仍然存在**（如把重婚罪法条说成第 313 条，实际为《刑法》第 258 条）→ SFT / DPO 能改"形式与风格"，改不了"事实"。

**取舍**：用小模型 + 小数据先把全链路跑通、把问题讲透，而非刷 SOTA 分数。

**下一步**：① 全量训练 / 更大底模；② 用强模型蒸馏更难的偏好数据；③ 引入真正的 LLM-as-judge + RAGAS 测忠实度与引用正确性；④ RAG 生产化（向量库 + 混合检索 + rerank + 检索不到则拒答）从根上压制法条幻觉。

---

## ⚠️ 诚实提示

- 本项目用**小模型 + 小数据**先跑通全链路，重在**链路完整 + 你能讲透**，不是刷 SOTA 分数。
- trl / peft 的 API 在不同版本略有差异；若某行报错，**读报错信息**对照官方文档微调参数——这本身就是后训练工程师的日常能力。
- "跑通"只是 60 分。把每个 notebook 里的【动手改一改】和【想一想】做掉，才能在面试里扛住追问。
