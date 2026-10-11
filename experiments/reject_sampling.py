# -*- coding: utf-8 -*-
"""拒绝采样:用 SFT 模型自己的输出造 on-policy 负样本。

【为什么规则劣化从根上就不对】
DPO 的损失是压低 rejected 的对数概率、抬高 chosen 的。但规则劣化造的 rejected
(「《民法典》第9999条」这种字符串)**模型本来就不会生成** —— 它在模型输出分布里的
概率质量本来就接近 0,压它等于没压。而模型真正会犯的错(它自己写出来的那些错法条)
从来没进过 rejected,所以从来没被压低过。

这比"负样本太假"更要害:不是难度问题,是**压错了对象**(off-policy)。
也解释了 05 的实测为什么是那个样子 ——
  · 复读治好了(25%→6%):corrupt_repeat 造的复读形态跟模型真实的复读几乎一样,
    恰好撞上了 on-policy;
  · 法条幻觉没治住(36%→39%):模型真实的幻觉是"引真实条号但内容错配",
    而规则造的是"第9999条",两者形态完全不同。

【本脚本做什么】
    对每个训练问题,让 sft_merged 采样生成 K 条回答(temperature 采样,不是贪心)
      → 用 legal_check 给每条打"硬伤分"
      → 挑最差的那条当 rejected(真实答案当 chosen)
      → 只保留确实有硬伤的(引用错误 / 复读 / 跑题),没硬伤的问题跳过

这就是 Llama 2/3 用的 rejection sampling:模型自己生成候选,用一个判别器筛。
区别是它们筛"最好的"拿去 SFT,这里筛"最差的"拿去当 DPO 负样本。

【用法】
    python3 experiments/reject_sampling.py            # 全量 600 条
    python3 experiments/reject_sampling.py 40         # 先跑 40 条试试
产物:data/dpo_onpolicy.json(prompt/chosen/rejected)+ outputs/eval/onpolicy_stats.json
"""
import os, sys, json, re, time, random
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import legal_check as L

DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"

K          = 3      # 每题采样几条(取最差的一条)
BATCH      = 8      # 实测 batch=8 时 2.1 s/条,batch=4 是 3.0 s/条
MAX_NEW    = 320    # rejected 不需要很长;320 已能暴露引用错误和复读
TEMP, TOPP = 0.9, 0.95
SEED       = 42

SYSTEM_PROMPT = ("你是专业的中文法律咨询助手。请针对用户的法律问题，给出准确、有依据、条理清晰的解答；"
                 "涉及具体法条时尽量说明，不要编造不存在的法条或事实。")


def load_sources(n=None):
    """复刻 02 的划分与清洗,拿到同一批 dpo_src(保证与规则版可对照)"""
    random.seed(SEED)
    pool = json.load(open(os.path.join(DATA, "qa_pool.json"), encoding="utf-8"))
    random.shuffle(pool)
    train_pool = pool[150:]

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
    return clean[:(n or 600)]


def badness(q, ans, db):
    """硬伤分:数越大越差。优先级 = 引用错误 > 复读 > 跑题。

    法条幻觉排第一,因为它正是当前 DPO **治不了**的那一项 ——
    造负样本要对着没治好的病造,不是对着已经治好的造。
    """
    o = L.objective_v2(q, ans, db)
    rep = L.repetition_rate(ans)        # objective_v2 只给 bool,这里要连续值才能排序
    return (o["引用错误"] * 10          # 引错法条:最想治的
            + rep * 3                   # 复读
            + (1 - o["切题度"]) * 2,    # 跑题
            o, rep)


@torch.no_grad()
def gen_batch(model, tok, questions):
    texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True) for q in questions]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                         temperature=TEMP, top_p=TOPP, pad_token_id=tok.pad_token_id)
    n_in = enc["input_ids"].shape[1]
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    n_src = int(sys.argv[1]) if len(sys.argv) > 1 else None
    src = load_sources(n_src)
    db  = L.load_law_db(os.path.join(DATA, "stard_corpus.jsonl"), verbose=False)

    sft = os.path.join(OUT, "sft_merged")
    tok = AutoTokenizer.from_pretrained(sft)
    tok.padding_side = "left"                    # 生成必须左填充,否则续写会接在 pad 后面
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(sft, dtype=torch.float32).to(DEV).eval()

    print(f"拒绝采样:{len(src)} 题 × K={K} 条 / batch={BATCH} / max_new={MAX_NEW}")
    print(f"预计 {len(src)*K*2.1/60:.0f} 分钟\n", flush=True)

    torch.manual_seed(SEED)
    rows, stats, t0 = [], [], time.time()
    for i in range(0, len(src), BATCH):
        chunk = src[i:i + BATCH]
        qs = [x["question"] for x in chunk]
        cands = [[] for _ in chunk]
        for _ in range(K):                        # 同一批问题采样 K 次
            for j, g in enumerate(gen_batch(model, tok, qs)):
                cands[j].append(g)

        for x, cs in zip(chunk, cands):
            scored = [(badness(x["question"], c, db), c) for c in cs if len(c.strip()) > 20]
            if not scored:
                continue

            # 【为什么不无脑取最差的】首版取 max(badness),结果 rejected 中位 400 字、
            # chosen 只有 278 字 —— 长度差 44%,和 corrupt_truncate 一个量级。
            # 那样等于把好不容易消掉的表面特征又请了回来:模型学"长的是坏的"就能分对。
            # 改成两段式:先筛"够差"的候选,再在其中选**长度最接近 chosen** 的。
            # 牺牲一点"最差",换掉一个最强的表面特征,值得。
            def is_bad(t):
                (_, o, r) = t[0]
                return o["引用错误"] > 0 or r > 0.15 or o["切题度"] < 0.25

            qualified = [t for t in scored if is_bad(t)]
            if qualified:
                (score, obj, rep), worst = min(
                    qualified, key=lambda t: abs(len(t[1]) - len(x["answer"])))
            else:                                  # 都不够差 → 取最差的,交给 keep 过滤
                (score, obj, rep), worst = max(scored, key=lambda t: t[0][0])
            keep = obj["引用错误"] > 0 or rep > 0.15 or obj["切题度"] < 0.25
            stats.append({"q": x["question"], "score": round(score, 3), "keep": bool(keep),
                          "引用错误": obj["引用错误"], "核实正确": obj["核实正确"],
                          "复读率": round(rep, 3), "切题度": round(obj["切题度"], 3),
                          "len_rej": len(worst), "len_cho": len(x["answer"])})
            if keep:
                rows.append({
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": x["question"]}],
                    "chosen": [{"role": "assistant", "content": x["answer"]}],
                    "rejected": [{"role": "assistant", "content": worst}]})

        done = min(i + BATCH, len(src))
        el = time.time() - t0
        print(f"  {done}/{len(src)}  产出 {len(rows)} 条  "
              f"用时 {el/60:.1f}min  剩余约 {el/done*(len(src)-done)/60:.0f}min", flush=True)

    json.dump(rows, open(os.path.join(DATA, "dpo_onpolicy.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    json.dump({"K": K, "max_new": MAX_NEW, "temperature": TEMP,
               "n_src": len(src), "n_kept": len(rows), "detail": stats},
              open(os.path.join(OUT, "eval", "onpolicy_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    import statistics as st
    kept = [s for s in stats if s["keep"]]
    print("\n" + "=" * 62)
    print(f"产出 {len(rows)}/{len(src)} 条 on-policy 负样本 ({len(rows)/max(1,len(src)):.0%})")
    print("=" * 62)
    if kept:
        print(f"  含引用错误      {sum(1 for s in kept if s['引用错误']>0):>4} 条")
        print(f"  复读率>0.15     {sum(1 for s in kept if s['复读率']>0.15):>4} 条")
        print(f"  切题度<0.25     {sum(1 for s in kept if s['切题度']<0.25):>4} 条")
        _lr = st.mean(s["len_rej"] for s in kept); _lc = st.mean(s["len_cho"] for s in kept)
        _per = st.mean(abs(s["len_rej"] - s["len_cho"]) / max(1, s["len_cho"]) for s in kept)
        _lgr = sum(1 for s in kept if s["len_rej"] > s["len_cho"]) / len(kept)
        print(f"  rejected 平均 {_lr:.0f} 字 | chosen 平均 {_lc:.0f} 字")
        print(f"  长度特征:  系统性偏差 {(_lr-_lc)/_lc:+.1%}"
              f" | 逐条离散 {_per:.1%} | rejected 更长占 {_lgr:.0%}")
        print("\n  ⚠️ 这两个数含义完全不同,别混:")
        print("     系统性偏差(有符号) —— 方向稳定才能被模型学成判别规则(\"长的是坏的\");")
        print("     逐条离散(无符号)   —— 再大也没关系,只要方向随机(更长占比≈50%)就学不出规则。")
        print("     首版无脑取最差时系统性偏差是 +44%,靠'够差的里面挑长度最接近的'压到了 ≈0。")
    print("\n已存 data/dpo_onpolicy.json + outputs/eval/onpolicy_stats.json")
