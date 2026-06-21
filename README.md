# 中文法律问答大模型后训练：SFT + DPO

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-success)
![License](https://img.shields.io/badge/License-MIT-green)

在复旦 DISC-Law 真实中文法律问答数据上，对 Qwen2.5 完成 SFT + DPO 后训练对齐，建立 base / +SFT / +SFT+DPO 三档消融评测，并针对评测中暴露的法条幻觉补充了 RAG 检索增强 baseline。

> 6 个 notebook 串起完整后训练链路。用小模型 + 小数据验证方法论与可复现性，不追求 SOTA。

```mermaid
flowchart LR
  A[数据构造 SFT/DPO] --> B[LoRA SFT] --> C[DPO 偏好对齐] --> D[三档消融评测] --> E[RAG 检索增强]
```

## 主要工作

- 数据清洗去重与偏好数据构造（用 5 类劣化规则把真实答案改坏，生成 DPO 偏好对）。
- LoRA SFT → DPO 对齐 → 合并导出完整模型。
- 字符级 BLEU + 规则 judge + badcase 的三档消融评测，并结合结果做问题分析。

## 结果

评测集与训练集隔离（base / +SFT / +SFT+DPO）：

| 模型 | 字符 BLEU | 规则 judge |
|---|---|---|
| base | 0.051 | 0.787 |
| +SFT | 0.058 | 0.779 |
| +SFT+DPO | 0.065 | 0.752 |

loss 曲线与消融柱状图见 [`outputs/figures/`](outputs/figures)。

## 技术栈

Python · PyTorch · Hugging Face Transformers / PEFT / TRL · ChromaDB · bge embedding · Qwen2.5

## 项目结构

```
notebooks/   01 数据探查与清洗 · 02 SFT/DPO 数据构造 · 03 LoRA SFT
             04 DPO 对齐 · 05 消融评测 · 06 RAG 增强
data/        seed_clauses.jsonl（离线兜底样本）、clauses.jsonl（RAG 法条）
outputs/     ablation_table.md、figures/、eval/
```

## 运行

```bash
pip install -r requirements.txt
# 按顺序运行 notebooks/ 下 01 → 06
# 每个 notebook 顶部配置区可切 run_mode = "smoke"(快) / "full"(完整)
# 国内下载模型/数据慢可设：export HF_ENDPOINT=https://hf-mirror.com
```

默认底模 Qwen2.5-0.5B-Instruct（单卡 / Apple MPS 可跑），可切 1.5B。

## 数据集

- 训练数据来自 DISC-Law-SFT（复旦 FudanDISC 开源）：<https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT>
- `notebooks/01` 首次运行自动下载样本（默认 2000 条，走 hf-mirror），下不动时回退内置 `data/seed_clauses.jsonl`。
- 数据集版权归原作者，下载数据已在 `.gitignore` 忽略、不随仓库分发。

## 结论

在 DISC-Law 中文法律问答上对 Qwen2.5 完成 SFT + DPO 后训练全流程：数据清洗去重与偏好数据构造 → LoRA 微调 → DPO 偏好对齐 → 三档消融评测；并针对法条幻觉补充 RAG 检索增强 baseline。

- SFT：模型学会专业法律口吻与"引用法条"的输出格式，BLEU 0.051 → 0.058。
- DPO：BLEU 进一步升到 0.065，rewards / accuracies 收敛至约 1.0。

几点分析：

- judge 分数逐档下降，分析后发现是规则 judge 在奖励"关键词堆砌"，说明评测指标的设计往往比训练本身更关键，单一指标并不可靠。
- DPO accuracies 收敛到约 1.0，是因为偏好对用劣化规则构造、过于容易区分；真实场景需要更"难"的 rejected。
- badcase 显示法条事实幻觉仍然存在（如把重婚罪法条说成第 313 条，实际为《刑法》第 258 条），说明 SFT / DPO 能改输出的形式与风格，但改不了事实层面的错误。

## 局限与展望

- 当前为小模型 + 小数据的链路验证；后续可做全量训练 / 更大底模。
- 用更强模型蒸馏更"难"的偏好数据；引入真正的 LLM-as-judge + RAGAS 评估忠实度与引用正确性。
- RAG 生产化（向量库 + 混合检索 + rerank + 检索不到则拒答）以缓解法条幻觉。

## License

[MIT](LICENSE)
