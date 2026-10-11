# -*- coding: utf-8 -*-
"""checkpoint 序列评测:DPO 到底该停在第几步?

【为什么要跑】
本轮 DPO(2126 对,24% on-policy)跑满 266 步,完整曲线显示两个矛盾信号:

    accuracies  0.700 起,爬到 0.957        —— 看起来一路向好
    loss        第 220 步见底 0.1813,末值 0.2821(反弹 56%)
    chosen 奖励 −0.00 一路走到 **−3.57**    —— likelihood displacement

`chosen 奖励` 为负意味着**模型给"好答案"的概率比训练前还低**。margins 涨到 5~6
不是靠抬高 chosen,是靠把 rejected 压得更狠。这会导致输出退化(变短、变保守)——
上一轮实测的"字数 335→221、引用准确率 34%→20%"很可能就是它。

**训练指标救不了这个判断,必须看下游。** 本脚本对三个 checkpoint 跑同一套评测,
用 05 的客观指标回答:最优点在哪、是不是真的该早停。

【设计】
· 三档:150 / 200 / 266 步(150 是 loss 较低且 chosen 未过度下跌的位置)
· base 只加载一次,用 PEFT 的多 adapter 机制切换,省时省显存
· fp32 + 贪心 + max_new_tokens=512,与 05 口径一致
· batch=8 批量生成(05 是逐条)——**档位之间完全可比**,与 05 的绝对值可能有极小差异

【用法】
    python3 experiments/checkpoint_sweep.py
产物:outputs/eval/checkpoint_sweep.json
"""
import os, sys, json, time, statistics as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import legal_check as L

DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
STEPS_TO_TEST = ["sft", 250, 266]
BATCH, MAX_NEW = 8, 512


@torch.no_grad()
def gen_batch(model, tok, systems, users):
    texts = [tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True)
             for s, u in zip(systems, users)]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    n = enc["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]


def evaluate(preds, items, db):
    obs = [L.objective_v2(it["user"], p, db) for it, p in zip(items, preds)]
    ok  = sum(o["核实正确"] for o in obs)
    bad = sum(o["引用错误"] for o in obs)
    return {
        "平均字数":   round(st.mean(len(p) for p in preds)),
        "有引用率":   round(sum(o["有引用"] for o in obs) / len(obs), 3),
        "引用总数":   ok + bad,
        "核实正确":   ok,
        "引用准确率": round(ok / max(1, ok + bad), 3),
        "法条幻觉率": round(st.mean(o["法条幻觉"] for o in obs), 3),
        "复读率":     round(st.mean(o["复读"] for o in obs), 3),
        "跑题率":     round(st.mean(o["跑题"] for o in obs), 3),
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    sft_eval = json.load(open(os.path.join(DATA, "sft_eval.json"), encoding="utf-8"))
    items = [{"system": m["messages"][0]["content"], "user": m["messages"][1]["content"]}
             for m in sft_eval]
    db = L.load_law_db(os.path.join(DATA, "stard_corpus.jsonl"), verbose=False)

    SFT = os.path.join(OUT, "sft_merged")
    tok = AutoTokenizer.from_pretrained(SFT)
    tok.padding_side = "left"                 # 生成必须左填充
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(SFT, dtype=torch.float32).to(DEV).eval()

    # 【坑 2】PeftModel.from_pretrained(base, ...) **原地改造 base** ——
    # `base is peft_model.base_model.model` 为 True,包装之后再用 base 做前向,
    # adapter 依然生效。首版就是这么错的:"sft" 档测出来和 ck250 八个指标一模一样。
    # 正确做法是 `with model.disable_adapter():`,见下面的生成循环。
    #
    # 【教训】04 里 save_total_limit=2 → checkpoint-150/200 早被自动删了。
    # 想留完整序列做事后分析,必须提前把这个值调大,或另存一份。
    # 现在只剩 250/266(相差 16 步),因此补一档 "sft"(不启用任何 adapter)作基准 ——
    # 它同时是**口径校验**:测出来若接近 05 的 SFT 档(字数 335 / 引用准确率 34%),
    # 说明这里的 batch 生成与 05 的逐条生成可比。
    avail = [x for x in STEPS_TO_TEST if x == "sft"
             or os.path.isdir(os.path.join(OUT, "dpo_lora", f"checkpoint-{x}"))]
    print("实测档位:", avail)
    model = None
    for x in avail:
        if x == "sft":
            continue
        path = os.path.join(OUT, "dpo_lora", f"checkpoint-{x}")
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=f"ck{x}")
        else:
            model.load_adapter(path, adapter_name=f"ck{x}")
    model.eval()
    STEPS_TO_TEST = avail

    print(f"评测 {len(items)} 条 × {len(STEPS_TO_TEST)} 档 (fp32/贪心/max_new={MAX_NEW})")
    print(f"预计 {len(items)*len(STEPS_TO_TEST)*3.4/60:.0f} 分钟\n", flush=True)

    res, preds_all = {}, {}
    import contextlib

    def run_one(s):
        t0, preds = time.time(), []
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            preds += gen_batch(model, tok,
                               [c["system"] for c in chunk], [c["user"] for c in chunk])
            if (i // BATCH) % 5 == 0:
                print(f"  {s}: {min(i+BATCH,len(items))}/{len(items)}"
                      f"  {(time.time()-t0)/60:.1f}min", flush=True)
        return preds, time.time() - t0

    for s in STEPS_TO_TEST:
        if s == "sft":
            ctx = model.disable_adapter()          # ← 关键:临时关掉全部 adapter
        else:
            model.set_adapter(f"ck{s}")
            ctx = contextlib.nullcontext()
        with ctx:
            preds, el = run_one(s)
        t0 = time.time() - el
        res[s] = evaluate(preds, items, db)
        preds_all[s] = preds
        print(f"  ✅ ck{s} 完成 {(time.time()-t0)/60:.1f} 分  {res[s]}\n", flush=True)

    # ---------- 对比表 ----------
    keys = ["平均字数", "有引用率", "引用总数", "核实正确", "引用准确率",
            "法条幻觉率", "复读率", "跑题率"]
    print("=" * 72)
    print(f"{'指标':<12}" + "".join(f"{'ck'+str(s):>12}" for s in res))
    print("-" * 72)
    for k in keys:
        row = "".join(f"{res[s][k]:>12}" for s in res)
        print(f"{k:<12}{row}")
    print()
    print("上一轮(纯规则 450 步)参照: 字数 221 | 有引用率 45% | 引用准确率 20% | 幻觉率 39%")
    print("SFT 档参照:                字数 335 | 有引用率 54% | 引用准确率 34% | 幻觉率 36%")
    print()
    print("判读:")
    print("  · 若 ck150 的引用准确率/字数明显优于 ck266 → 证实 likelihood displacement 导致退化,")
    print("    且 **chosen 奖励是比 loss 更好的早停判据**;")
    print("  · 若三档接近 → 说明 266 步还没到损害下游的程度,早停收益有限。")
    print("  ⚠️ 看引用准确率必须同时看『引用总数』——准确率上升可能只是『少说少错』。")

    # ---------- 自检 ----------
    # 首版没有这一步,导致 "sft" 和 ck250 八个指标全同却没被发现,差点把错误结果当真。
    sigs = {str(k): tuple(v[x] for x in keys) for k, v in res.items()}
    dup = [(a, b) for i, a in enumerate(sigs) for b in list(sigs)[i+1:] if sigs[a] == sigs[b]]
    if dup:
        print("\n❌ 自检失败:以下档位结果完全相同,说明模型没被真正切换 →", dup)
    else:
        print("\n✅ 自检通过:各档位结果互不相同")

    json.dump({"配置": {"batch": BATCH, "max_new_tokens": MAX_NEW, "dtype": "fp32"},
               "结果": {str(k): v for k, v in res.items()},
               "生成": {str(k): v for k, v in preds_all.items()}},
              open(os.path.join(OUT, "eval", "checkpoint_sweep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已存 outputs/eval/checkpoint_sweep.json")
