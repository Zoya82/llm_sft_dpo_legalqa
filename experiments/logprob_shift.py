# -*- coding: utf-8 -*-
"""直接测量 likelihood displacement:DPO 后模型给"好答案"的概率降了多少。

【为什么不用 checkpoint 序列去验】
本来打算用 ck150 vs ck266 的下游指标来验证 displacement 的影响,但那个设计有两个问题:

  ① **对比点选错了**。chosen 奖励在 step150 已经是 −3.00、step266 才 −3.57 ——
     退化的 84% 发生在前 150 步,这两点之间只覆盖了最后一小段。
  ② **归因不干净**。训练步数同时改变两件事:学到的对齐更多(正向)、chosen 概率被压低
     (负向)。下游指标是净结果,分不清哪个在起作用。

【这个脚本怎么做】
回到机制本身 —— displacement 的定义就是"模型给好答案的概率下降",那就直接测它:

    对 148 条评测集的**参考答案**,算 log P(参考答案 | 问题)
      sft_merged  → DPO 之前
      ck250/266   → DPO 之后

不生成、只做一次前向,成本从 25 分钟降到 3 分钟,而且**直接对应机制**。

评测集的参考答案模型没见过,所以这测的是泛化意义上的"好答案概率",
不是训练集上的记忆。

【指标】
· 平均每 token 的 log-prob(消除答案长度差异)
· 逐条配对差值(同一条样本前后对比,比两组均值相减更灵敏)
· 下降条数占比 —— **方向一致性**才是判据,不是均值差的大小
  (这条是踩过两次坑之后的教训:有符号的量和无符号的量含义完全不同)

【用法】
    python3 experiments/logprob_shift.py
产物:outputs/eval/logprob_shift.json
"""
import os, sys, json, time, statistics as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
CKPTS = ["sft", 250, 266]


@torch.no_grad()
def seq_logprob(model, tok, system, user, answer):
    """返回 (答案总 log-prob, 每 token 平均 log-prob, token 数)"""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    f_ids = tok(prompt + answer, return_tensors="pt").input_ids
    n_p = p_ids.shape[1]
    if f_ids.shape[1] <= n_p:
        return None
    logits = model(f_ids.to(DEV)).logits            # 一次前向,不生成
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = f_ids[0, 1:].to(DEV)
    tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    ans_lp = tok_lp[n_p - 1:]                       # 只取答案部分
    return ans_lp.sum().item(), ans_lp.mean().item(), ans_lp.numel()


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    sft_eval = json.load(open(os.path.join(DATA, "sft_eval.json"), encoding="utf-8"))
    items = [{"system": m["messages"][0]["content"],
              "user":   m["messages"][1]["content"],
              "answer": m["messages"][2]["content"]} for m in sft_eval]

    SFT = os.path.join(OUT, "sft_merged")
    tok = AutoTokenizer.from_pretrained(SFT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.float32).to(DEV).eval()

    avail = [c for c in CKPTS if c == "sft"
             or os.path.isdir(os.path.join(OUT, "dpo_lora", f"checkpoint-{c}"))]
    model = None
    for c in avail:
        if c == "sft":
            continue
        p = os.path.join(OUT, "dpo_lora", f"checkpoint-{c}")
        if model is None:
            model = PeftModel.from_pretrained(base, p, adapter_name=f"ck{c}")
        else:
            model.load_adapter(p, adapter_name=f"ck{c}")
    if model is not None:
        model.eval()

    print(f"测 {len(items)} 条参考答案的 log-prob × {len(avail)} 档(只前向,不生成)\n", flush=True)

    res = {}
    for c in avail:
        gm = base
        if c != "sft":
            model.set_adapter(f"ck{c}")
            gm = model
        t0, rows = time.time(), []
        for i, it in enumerate(items):
            r = seq_logprob(gm, tok, it["system"], it["user"], it["answer"])
            if r:
                rows.append({"总": r[0], "均": r[1], "tok": r[2]})
            if (i + 1) % 50 == 0:
                print(f"  {c}: {i+1}/{len(items)}  {(time.time()-t0)/60:.1f}min", flush=True)
        res[str(c)] = rows
        print(f"  ✅ {c}: 平均每token log-prob {st.mean(x['均'] for x in rows):.4f}"
              f"  ({(time.time()-t0)/60:.1f} 分)\n", flush=True)

    # ---------------- 对比 ----------------
    print("=" * 72)
    print(f"{'档位':<10}{'平均每token':>14}{'中位':>12}{'总log-prob均值':>16}")
    print("-" * 72)
    for c in avail:
        r = res[str(c)]
        print(f"{str(c):<10}{st.mean(x['均'] for x in r):>14.4f}"
              f"{st.median(x['均'] for x in r):>12.4f}"
              f"{st.mean(x['总'] for x in r):>16.1f}")

    if "sft" in [str(c) for c in avail]:
        b = res["sft"]
        print("\n逐条配对对比(同一样本前后差,比均值相减灵敏):")
        print(f"{'档位':<10}{'平均Δ/token':>14}{'下降条数':>10}{'占比':>9}{'判读':>22}")
        print("-" * 72)
        for c in avail:
            if str(c) == "sft":
                continue
            a = res[str(c)]
            d = [x["均"] - y["均"] for x, y in zip(a, b)]
            down = sum(1 for v in d if v < 0)
            ratio = down / len(d)
            verdict = ("方向一致 → 系统性下降" if ratio >= 0.8 else
                       "方向随机 → 无系统性变化" if 0.35 < ratio < 0.65 else "部分下降")
            print(f"{str(c):<10}{st.mean(d):>+14.4f}{down:>10}{ratio:>9.0%}{verdict:>22}")
        print("\n判读要点:")
        print("  · Δ 显著为负 **且** 下降条数占比 ≥80% → likelihood displacement 被直接证实")
        print("  · 每 token log-prob 降 0.1 ≈ 该 token 的概率变为原来的 exp(-0.1)=90%")
        print("  · 若方向随机(占比≈50%)→ 只是数值扰动,不是系统性压低")

    json.dump({"档位": [str(c) for c in avail], "逐条": res},
              open(os.path.join(OUT, "eval", "logprob_shift.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已存 outputs/eval/logprob_shift.json")
