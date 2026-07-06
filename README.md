# 中文法律问答：Qwen2.5 后训练（SFT + DPO）与评测（DISC-Law）

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-success)
![License](https://img.shields.io/badge/License-MIT-green)

基于 Qwen2.5-0.5B-Instruct 在复旦 DISC-Law 中文法律问答数据上完成 LoRA SFT 与 DPO 偏好对齐，设置 base / +SFT / +SFT+DPO 三档消融，用字符级 BLEU 与规则 judge 评估，并针对评测中暴露的法条幻觉补充 RAG 检索增强 baseline。

## 结果（评测集与训练集隔离）

| 实验 | 字符 BLEU | 规则 judge |
|---|---|---|
| base | 0.051 | 0.787 |
| +SFT | 0.058 | 0.779 |
| +SFT+DPO | **0.065** | 0.752 |

judge 逐档下降源于规则 judge 奖励"关键词堆砌"，单一指标不可靠；loss 曲线与消融对比图见 [`outputs/figures/`](outputs/figures)，badcase 见 [`outputs/eval/`](outputs/eval)。

## 方法

数据清洗去重 → SFT/DPO 数据构造（5 类劣化规则生成偏好对）→ LoRA SFT → DPO 对齐 → 三档消融评测（BLEU + 规则 judge + badcase）→ RAG 检索增强对比。

## 结构

```
notebooks/  01 数据探查与清洗 · 02 SFT/DPO 数据构造 · 03 LoRA SFT
            04 DPO 对齐 · 05 消融评测 · 06 RAG 增强
data/       seed_clauses.jsonl（离线兜底样本）· clauses.jsonl（RAG 法条）
outputs/    ablation_table.md · figures/ · eval/
```

## 运行

```bash
pip install -r requirements.txt
# 按顺序运行 notebooks/ 01 → 06
# 每个 notebook 顶部可切 run_mode = "smoke"(快) / "full"(完整)
```

底模 Qwen2.5-0.5B-Instruct（Apple MPS / 单卡可跑），可切 1.5B。技术栈：Transformers / PEFT / TRL / ChromaDB / bge embedding。

## 数据集

DISC-Law-SFT（复旦 FudanDISC 开源），版权归原作者，未随仓库分发（见 .gitignore）。
`notebooks/01` 首次运行自动下载样本（默认 2000 条），下不动时回退内置 `data/seed_clauses.jsonl`。

## License

[MIT](LICENSE)
