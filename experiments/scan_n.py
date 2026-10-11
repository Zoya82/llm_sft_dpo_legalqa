# -*- coding: utf-8 -*-
"""扫描 n=2..6,看正负例的分离度,给 _char_overlap 的 n 定一个有依据的取值。

正例:拿真实条文当作"模型的引用"(逐字照抄)——应该判 ok
负例:条号和内容故意错配——应该判 misattributed
转述例:对真实条文做轻度改写(加"的"、换同义词、改标点)——这是最容易被误杀的一类
"""
from pathlib import Path
import sys, random, statistics as st
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import legal_check as L

DB = L.load_law_db(str(Path(__file__).resolve().parents[1] / "data/stard_corpus.jsonl"),
                   verbose=False)

def overlap(a, b, n):
    """和 legal_check._char_overlap 同一套逻辑,n 可调"""
    if len(a) < n or len(b) < n:
        return 0.0
    ga = [a[i:i+n] for i in range(len(a)-n+1)]
    cb = Counter(b[i:i+n] for i in range(len(b)-n+1))
    hit = 0
    for g in ga:
        if cb.get(g, 0) > 0:
            cb[g] -= 1
            hit += 1
    return hit / len(ga)

def paraphrase(s, rng):
    """模拟模型转述:随机替换几个同义词 + 删几个"的" + 全角括号换半角"""
    for a, b in [("应当", "应该"), ("可以", "能够"), ("但是", "但"), ("或者", "或"),
                 ("（", "("), ("）", ")"), ("；", ";")]:
        s = s.replace(a, b)
    chars = list(s)
    drop = [i for i, c in enumerate(chars) if c == "的"]
    for i in rng.sample(drop, min(2, len(drop))):
        chars[i] = ""
    return "".join(chars)

LAWS = ["刑法", "民法典", "公司法", "仲裁法", "劳动合同法", "道路交通安全法"]
rng = random.Random(0)

# 采样:每部法律取若干条正条(sub=0)
samples = []
for law in LAWS:
    keys = [k for k in DB[law] if k[1] == 0 and len(DB[law][k]) >= 30]
    for k in rng.sample(keys, min(25, len(keys))):
        samples.append((law, k))

print(f"样本:{len(samples)} 条真实条文,来自 {len(LAWS)} 部法律\n")

rows = []
for n in range(2, 7):
    pos, para, neg = [], [], []
    for law, k in samples:
        truth = DB[law][k]
        claim = truth[:70]                       # 正例:逐字照抄前 70 字
        pos.append(overlap(claim, truth, n))
        para.append(overlap(paraphrase(claim, rng), truth, n))   # 转述例
    # 负例:条号与内容错配
    for law in LAWS:
        keys = [k for k in DB[law] if k[1] == 0 and len(DB[law][k]) >= 30]
        A = rng.sample(keys, min(25, len(keys)))
        B = rng.sample(keys, min(25, len(keys)))
        for ka, kb in zip(A, B):
            if ka != kb:
                neg.append(overlap(DB[law][kb][:70], DB[law][ka], n))
    rows.append({
        "n": n,
        "pos_min": min(pos), "pos_med": st.median(pos),
        "para_min": min(para), "para_med": st.median(para),
        "neg_max": max(neg), "neg_p95": sorted(neg)[int(len(neg)*0.95)], "neg_med": st.median(neg),
        "gap": min(para) - max(neg),              # 分离度:最难的正例 - 最像的负例
    })

print(f"{'n':>2} | {'正例(逐字)':>16} | {'转述例':>16} | {'负例(错配)':>22} | {'分离度':>8}")
print(f"{'':>2} | {'最低':>7}{'中位':>8} | {'最低':>7}{'中位':>8} | {'最高':>7}{'p95':>7}{'中位':>7} | {'转述最低-负例最高':>8}")
print("-" * 82)
for r in rows:
    print(f"{r['n']:>2} | {r['pos_min']:>7.3f}{r['pos_med']:>8.3f} | "
          f"{r['para_min']:>7.3f}{r['para_med']:>8.3f} | "
          f"{r['neg_max']:>7.3f}{r['neg_p95']:>7.3f}{r['neg_med']:>7.3f} | {r['gap']:>8.3f}")

print("\n【短条文覆盖率】不同 n 下,有多少条文因为太短而无法比对(直接返回 0):")
alllen = [len(v) for law in DB.values() for v in law.values()]
for n in range(2, 7):
    bad = sum(1 for x in alllen if x < n)
    print(f"  n={n}: {bad} / {len(alllen)} 条 ({100*bad/len(alllen):.2f}%)")

best = max(rows, key=lambda r: r["gap"])
print(f"\n分离度最大的 n = {best['n']}(gap={best['gap']:.3f})")
