# 中文法律问答大模型后训练：SFT + DPO

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-success)
![TRL](https://img.shields.io/badge/TRL-SFT%20%7C%20DPO-orange)
![License](https://img.shields.io/badge/License-MIT-green)

在复旦 **DISC-Law** 真实中文法律问答数据上，对 **Qwen2.5** 完成 **SFT + DPO** 后训练对齐，建立 base / +SFT / +SFT+DPO 三档消融评测，并针对评测暴露的"法条幻觉"补充 **RAG 检索增强** baseline。

> 6 个 notebook 跑通完整后训练链路。用小模型 + 小数据验证**方法论与可复现性**，而非刷 SOTA。

```mermaid
flowchart LR
  A[数据构造 SFT/DPO] --> B[LoRA SFT] --> C[DPO 偏好对齐] --> D[三档消融评测] --> E[RAG 检索增强]
```

## ✨ 亮点

- **完整后训练链路**：数据清洗与偏好数据构造（5 类劣化规则生成 DPO 偏好对）→ LoRA SFT → DPO 对齐 → 合并导出。
- **可复现评测**：字符级 BLEU + 规则 LLM-as-judge + badcase 分析，三档消融对比。
- **批判性发现**（见[结论](#-结论)）：定位出评测指标"奖励关键词堆砌"的缺陷、DPO 偏好对过易区分、SFT/DPO 改不了法条事实幻觉，并用 RAG 验证改善路径。

## 📊 结果

评测集与训练严格隔离（base / +SFT / +SFT+DPO）：

| 模型 | 字符 BLEU | 规则 judge |
|---|---|---|
| base | 0.051 | 0.787 |
| +SFT | 0.058 | 0.779 |
| **+SFT+DPO** | **0.065** | 0.752 |

loss 曲线与消融柱状图见 [`outputs/figures/`](outputs/figures)。

## 🔧 技术栈

`Python` · `PyTorch` · `Hugging Face Transformers / PEFT / TRL` · `ChromaDB` · `bge embedding` · `Qwen2.5`

## 🗂 项目结构

```
notebooks/   01 数据探查与清洗 · 02 SFT/DPO 数据构造 · 03 LoRA SFT
             04 DPO 对齐 · 05 消融评测 · 06 RAG 增强
data/        seed_clauses.jsonl（离线兜底样本）、clauses.jsonl（RAG 法条）
outputs/     ablation_table.md、figures/、eval/
```

## ▶️ 运行

```bash
pip install -r requirements.txt
# 按顺序运行 notebooks/ 下 01 → 06
# 每个 notebook 顶部【配置区】可切 run_mode = "smoke"(快) / "full"(完整)
# 国内下载模型/数据慢可设：export HF_ENDPOINT=https://hf-mirror.com
```

默认底模 `Qwen2.5-0.5B-Instruct`（单卡 / Apple MPS 可跑），可切 1.5B。

## 📁 数据集与版权

- 训练数据来自 **DISC-Law-SFT**（复旦 FudanDISC 开源）：<https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT>
- `notebooks/01` 首次运行自动下载样本（默认 2000 条，走 hf-mirror），下不动时回退内置 `data/seed_clauses.jsonl`。
- 数据集版权归原作者，下载数据已在 `.gitignore` 忽略、不随仓库分发；使用请遵循其 License。

## 📝 结论

**做了什么**：在 DISC-Law 中文法律问答上对 Qwen2.5 完成 SFT + DPO 后训练全流程——数据清洗去重与偏好数据构造 → LoRA 微调 → DPO 偏好对齐 → 三档消融评测；并针对法条幻觉补充 RAG 检索增强 baseline。

- **SFT**：模型学会专业法律口吻与"引用法条"的输出格式；BLEU 0.051 → 0.058。
- **DPO**：BLEU 进一步升到 0.065，rewards / accuracies 收敛至约 1.0。

**关键发现（比分数更重要）**：
- judge 分数逐档**下降**，深挖发现是规则 judge 在奖励"关键词堆砌"——**评测指标设计往往比训练本身更关键**，单一指标不可靠。
- DPO accuracies 轻松收敛到约 1.0，是因为偏好对用"劣化规则"构造、过于易区分；真实场景需更"难"的 rejected。
- badcase 显示**法条事实幻觉仍然存在**（如把重婚罪法条说成第 313 条，实际为《刑法》第 258 条）→ SFT/DPO 能改"形式与风格"，改不了"事实"。

## 🚧 局限与展望

- 当前为小模型 + 小数据的链路验证；下一步：全量训练 / 更大底模；
- 用强模型蒸馏更"难"的偏好数据；引入真正的 LLM-as-judge + RAGAS 测忠实度与引用正确性；
- RAG 生产化（向量库 + 混合检索 + rerank + 检索不到则拒答）从根上压制法条幻觉。

## License

[MIT](LICENSE)
