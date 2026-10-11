# -*- coding: utf-8 -*-
"""生成盲评数据:SFT+RAG  vs  SFT+DPO+RAG。

【为什么这一组必须让人来判】
客观指标已经强烈倾向 SFT+RAG:

    SFT+RAG      引对 44 条 | 有引用率 46% | 幻觉 25% | 269 字
    SFT+DPO+RAG  引对 17 条 | 有引用率 22% | 幻觉 11% | 128 字

但幻觉率那一项是**"少说少错"**造成的(引用总数 94 vs 35),而 DPO 的优势
(不复读、不跑题)在人的主观感受里权重很高 —— 上一轮 DPO 就是**客观退步、
盲评却以 75% 大胜**。所以这一组不能只看客观数字。

【消偏设计,与既有三组盲评一致】
· 分层抽样:按两个回答的硬伤组合分层,保证难易分布不失真
· 随机打乱 A/B 位置,映射单独存盘,判完才揭晓
· 隐去模型名,只给问题 + 两个回答 + 各自的法条核验结果

【产物】
    outputs/eval/blind/rag_ab.md            ← 你看这个,逐条判
    outputs/eval/blind/rag_ab_mapping.json  ← 揭晓用,判完之前别打开

【用法】
    python3 experiments/make_blind_rag.py [条数,默认60]
"""
import os, sys, json, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import legal_check as L

DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs", "eval")
BLIND = os.path.join(OUT, "blind")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 20260805


def brief(ans, db):
    """给每个回答附上法条核验摘要 —— 判的时候不必自己查库"""
    cs = L.verify_citations(ans, db)
    lab = []
    for c in cs:
        mark = {"ok": "✓正确", "misattributed": "✗张冠李戴", "not_found": "✗条号不存在",
                "out_of_range": "✗条号越界", "repealed": "✗已废止",
                "uncertain": "?待定", "unverifiable": "?库中无此法"}.get(c["verdict"], c["verdict"])
        lab.append(f"{c['label']}{mark}")
    o = L.objective_v2("", ans, db)
    flaws = []
    if L.repetition_rate(ans) > 0.15:
        flaws.append("复读")
    if len(ans) < 60:
        flaws.append("过短")
    return {"字数": len(ans), "法条核验": lab or ["(未引用法条)"], "硬伤": flaws}


if __name__ == "__main__":
    db = L.load_law_db(os.path.join(DATA, "stard_corpus.jsonl"), verbose=False)
    ev = json.load(open(os.path.join(DATA, "sft_eval.json"), encoding="utf-8"))
    qs  = [m["messages"][1]["content"] for m in ev]
    ref = [m["messages"][2]["content"] for m in ev]
    a_dpo = json.load(open(os.path.join(OUT, "preds_rag.json"), encoding="utf-8"))
    a_sft = json.load(open(os.path.join(OUT, "preds_rag_sftbase.json"), encoding="utf-8"))
    assert len(qs) == len(a_dpo) == len(a_sft)

    # ---- 分层:按"两侧各自有无引用/有无硬伤"的组合,保证抽样不偏向某一类 ----
    strata = collections.defaultdict(list)
    for i in range(len(qs)):
        key = (bool(L.extract_citations(a_dpo[i])), bool(L.extract_citations(a_sft[i])))
        strata[key].append(i)

    rng = random.Random(SEED)
    picked = []
    for k, idxs in sorted(strata.items(), key=lambda kv: -len(kv[1])):
        take = max(1, round(len(idxs) * N / len(qs)))
        picked += rng.sample(idxs, min(take, len(idxs)))
    picked = sorted(picked)[:N]

    lines, mapping = [], {}
    lines.append("# 盲评:两个 RAG 版本\n")
    lines.append(f"共 {len(picked)} 条。两个回答都用了**同一套检索结果和同一个 prompt**,"
                 "唯一差别是生成模型。请只判**哪个回答更好**,不必猜是哪个模型。\n")
    lines.append("**判断建议**(与既有三组盲评口径一致):\n")
    lines.append("- 是否直接回答了问题;引用的法条是否正确(已附核验结果)\n")
    lines.append("- 有无复读、跑题、答非所问\n")
    lines.append("- **引用多但错几条** vs **几乎不引用但不出错** —— 哪个对提问的人更有用\n")
    lines.append("\n把结果记成一行,例如:`1A 2B 3A 4A ...`,发给我即可。\n\n---\n")

    for n, i in enumerate(picked, 1):
        flip = rng.random() < 0.5
        a_key, b_key = ("sft", "dpo") if flip else ("dpo", "sft")
        texts = {"dpo": a_dpo[i], "sft": a_sft[i]}
        mapping[str(n)] = {"idx": i, "A": a_key, "B": b_key}
        lines.append(f"\n## {n}. {qs[i]}\n")
        lines.append(f"> 参考答案(节选):{ref[i][:150]}…\n")
        for tag, key in (("A", a_key), ("B", b_key)):
            t = texts[key]
            b = brief(t, db)
            lines.append(f"\n**答案 {tag}** — {b['字数']} 字"
                         f"{' · 硬伤:' + '/'.join(b['硬伤']) if b['硬伤'] else ''}\n")
            lines.append(f"法条核验:{' | '.join(b['法条核验'])}\n\n")
            lines.append(f"{t}\n")
        lines.append("\n---\n")

    os.makedirs(BLIND, exist_ok=True)
    open(os.path.join(BLIND, "rag_ab.md"), "w", encoding="utf-8").write("".join(lines))
    json.dump(mapping, open(os.path.join(BLIND, "rag_ab_mapping.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    n_sft_at_a = sum(1 for v in mapping.values() if v["A"] == "sft")
    print(f"已生成 {len(picked)} 条 → outputs/eval/blind/rag_ab.md")
    print(f"分层数 {len(strata)} | SFT 版被排到 A 位的比例 {n_sft_at_a/len(mapping):.0%}"
          f"(应接近 50%,否则有位置偏差)")
    print("映射存于 rag_ab_mapping.json —— 判完再打开")
