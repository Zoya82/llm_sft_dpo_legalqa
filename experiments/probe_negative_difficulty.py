# -*- coding: utf-8 -*-
"""负样本难度探针:蒸馏出来的 rejected 到底有没有变难?

【要回答的问题】
04 的训练日志显示 rewards/accuracies **起步就 0.875**(随机猜 = 0.5),
说明规则劣化造的 rejected 太假、模型一眼能分辨。2.3 上了蒸馏式劣化,
但"数据变难"目前只有**表面统计**支撑(字符差异率 0.4~1.6% vs 25~99%),
还缺**模型视角**的证据 —— 模型是不是真的分不出来了。

【怎么测】
不跑完整训练(450 步要几小时),只跑前 40 步、取 4 个 log 点看 accuracies 的**起点**。
起点低 = 模型一开始分不出来 = 数据确实难。

【三组对照】
    rules_only  3600 对,纯规则劣化(上一轮的数据,已知起点 0.875)
    distilled   2220 对,规则 46% + 蒸馏 54%(2.3 的新数据)
    hard_only   1020 对,只有蒸馏难样本(**上限锚点**)

第三组是关键 —— 没有它就只能说"新数据比旧数据难",说不出"还能更难吗"。
(这条教训来自 LoRA 缩放扫描:当时缺 scale=0 锚点,把噪声当成了结论。)

【用法】
    python3 experiments/probe_negative_difficulty.py
产物:outputs/eval/negative_difficulty.json
"""
import os, sys, json, random, shutil, statistics as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import legal_check as L
import hard_negatives as D

DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
TMP  = os.path.join(ROOT, "outputs", "_probe_tmp")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_STEPS = 40          # 4 个 log 点(logging_steps=10),够看清起点
SEED      = 42

SYSTEM_PROMPT = ("你是专业的中文法律咨询助手。请针对用户的法律问题，给出准确、有依据、条理清晰的解答；"
                 "涉及具体法条时尽量说明，不要编造不存在的法条或事实。")


# ======================= 1. 重建三份数据 =======================
def build_datasets():
    """完全复刻 02 的构造逻辑(同样的 seed 和清洗),分别产出三份偏好数据"""
    import re
    random.seed(SEED)
    pool = json.load(open(os.path.join(DATA, "qa_pool.json"), encoding="utf-8"))
    random.shuffle(pool)
    train_pool = pool[150:]

    # --- 复刻 02 的清洗(只对训练侧) ---
    SPLIT = re.compile(r'(?:^|\n|(?<=[。；;！？]))\s*([二三四五六七八九十]{1,2}、)')
    END_OK = '。；;！？”")）】'

    def split_sections(ans):
        parts = SPLIT.split(ans)
        if len(parts) < 3:
            return ans.strip(), []
        head, secs = parts[0].strip(), []
        for k in range(1, len(parts) - 1, 2):
            body = parts[k + 1].strip()
            m = re.match(r'([^\n?？]{4,45}[?？])\s*(.+)', body, re.S) or \
                re.match(r'([^\n]{4,45})\n\s*(.+)', body, re.S)
            if m:
                secs.append((m.group(1).strip(' ?？'), m.group(2).strip()))
        return head, secs

    def dedup(t):
        seen, out = set(), []
        for p in re.split(r'(?<=[。；])', t):
            k = p.strip()
            if k and k not in seen:
                seen.add(k); out.append(p)
        return ''.join(out).strip()

    def usable(q, a):
        a, q = a.strip(), q.strip()
        return len(a) >= 40 and len(q) >= 4 and a[-1] in END_OK

    clean = []
    for x in train_pool:
        head, secs = split_sections(x['answer'])
        head = dedup(head)
        if usable(x['question'], head):
            clean.append({'question': x['question'], 'answer': head})
        for title, body in secs:
            body = dedup(body)
            q = title if re.search(r'吗|呢|怎么|如何|什么|哪些|多少|是否|能否', title) else title + '是什么'
            q = q.rstrip('?？') + '?'
            if usable(q, body):
                clean.append({'question': q, 'answer': body})
    dpo_src = clean[:600]

    # --- 劣化函数(与 02 一致) ---
    def c_truncate(a):    return a[: max(10, len(a)//2)]
    def c_hallucinate(a): return a[: max(20, len(a)//2)] + "综上，依据《民法典》第9999条，可直接主张三倍惩罚性赔偿。"
    def c_vague(a):       return "这个问题要具体情况具体分析，建议你咨询专业律师，以实际情况为准。"
    def c_flip(a):
        o = a
        for x, y in [("可以","不可以"),("应当","不应当"),("有权","无权"),
                     ("合法","违法"),("需要","不需要"),("属于","不属于")]:
            o = o.replace(x, y)
        return o if o != a else "（错误观点）" + a
    def c_offtopic(a):
        o = random.choice(pool)["answer"]
        return o if o.strip() != a.strip() else "(跑题)" + a
    def c_repeat(a):
        parts = [p for p in re.split(r'(?<=[。；;])', a) if len(p.strip()) > 10]
        if len(parts) < 2:
            return a + (a[:60] + "。") * 3
        k = len(parts)//2; n = max(5, min(len(parts), 8))
        o = "".join(parts[:k] + [parts[k]]*n + parts[k+1:])
        return o[:1100] if len(o) > 1100 else o

    EASY_ALL = [c_flip, c_hallucinate, c_truncate, c_vague, c_offtopic, c_repeat]
    EASY_SUB = [c_flip, c_hallucinate, c_truncate, c_vague, c_offtopic]

    db  = L.load_law_db(os.path.join(DATA, "stard_corpus.jsonl"), verbose=False)
    rng = random.Random(SEED)
    HARD = [("cite_shift", lambda a: D.corrupt_cite_shift(a, db, rng)),
            ("term_swap",  lambda a: D.corrupt_term_swap(a, rng)),
            ("degree",     lambda a: D.corrupt_degree(a, rng)),
            ("subject",    lambda a: D.corrupt_subject(a, rng))]

    rules, distilled, hard_only = [], [], []
    for x in dpo_src:
        ans = x["answer"]
        pr = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": x["question"]}]
        ch = [{"role": "assistant", "content": ans}]

        def add(bucket, bad):
            if bad and bad.strip() and bad.strip() != ans.strip():
                bucket.append({"prompt": pr, "chosen": ch,
                               "rejected": [{"role": "assistant", "content": bad}]})

        for fn in EASY_ALL:                       # 组1:纯规则,6 类全上
            add(rules, fn(ans))

        add(distilled, c_repeat(ans))             # 组2:repeat 保底 + 五选一 + 难样本全上
        add(distilled, rng.choice(EASY_SUB)(ans))
        for _, f in HARD:
            bad = f(ans)
            add(distilled, bad)
            add(hard_only, bad)                   # 组3:只要难样本

    for name, rows in [("rules_only", rules), ("distilled", distilled), ("hard_only", hard_only)]:
        json.dump(rows, open(os.path.join(TMP, f"{name}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"  {name:<12} {len(rows):>5} 对")
    return {"rules_only": len(rules), "distilled": len(distilled), "hard_only": len(hard_only)}


# ======================= 2. 短程训练取 accuracies =======================
def probe(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset

    sft = os.path.join(OUT, "sft_merged")
    tok = AutoTokenizer.from_pretrained(sft)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(sft, dtype=torch.float32).to(DEV)
    ds = load_dataset("json", data_files=os.path.join(TMP, f"{name}.json"), split="train")

    args = DPOConfig(
        output_dir=os.path.join(TMP, f"run_{name}"),
        max_steps=MAX_STEPS, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=5e-5, beta=0.1, logging_steps=10,
        save_strategy="no", max_length=768, gradient_checkpointing=True,
        dataloader_pin_memory=False, bf16=False, fp16=False, report_to="none",
        seed=SEED, data_seed=SEED,           # 三组用同一个抽样顺序种子,保证可比
    )
    tr = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                    processing_class=tok,
                    peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules="all-linear", task_type="CAUSAL_LM"))
    tr.train()
    logs = [{"step": l["step"], "acc": l["rewards/accuracies"], "margin": l["rewards/margins"],
             "loss": l["loss"]}
            for l in tr.state.log_history if "rewards/accuracies" in l]
    del model, tr
    import gc; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return logs


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    print("=" * 68)
    print("负样本难度探针:三组数据各跑前 40 步,看 rewards/accuracies 的起点")
    print("=" * 68)
    print("\n[1/2] 重建三份偏好数据")
    sizes = build_datasets()

    print(f"\n[2/2] 短程训练(各 {MAX_STEPS} 步)")
    res = {}
    for name in ["rules_only", "distilled", "hard_only"]:
        print(f"\n--- {name} ---", flush=True)
        res[name] = probe(name)

    print("\n" + "=" * 68)
    print("结果:rewards/accuracies(0.5=完全分不出,1.0=全分对)")
    print("=" * 68)
    steps = [l["step"] for l in res["rules_only"]]
    print(f"{'数据集':<14}{'对数':>7}" + "".join(f"{'第'+str(s)+'步':>10}" for s in steps))
    print("-" * 68)
    for name in ["rules_only", "distilled", "hard_only"]:
        row = "".join(f"{l['acc']:>10.3f}" for l in res[name])
        print(f"{name:<14}{sizes[name]:>7}{row}")
    print()
    print(f"{'数据集':<14}{'对数':>7}" + "".join(f"{'margin@'+str(s):>12}" for s in steps))
    print("-" * 68)
    for name in ["rules_only", "distilled", "hard_only"]:
        row = "".join(f"{l['margin']:>12.2f}" for l in res[name])
        print(f"{name:<14}{sizes[name]:>7}{row}")

    json.dump({"说明": "前 40 步 rewards/accuracies;起点越低=负样本越难",
               "max_steps": MAX_STEPS, "sizes": sizes, "logs": res},
              open(os.path.join(OUT, "eval", "negative_difficulty.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已存 outputs/eval/negative_difficulty.json")
    shutil.rmtree(TMP, ignore_errors=True)
