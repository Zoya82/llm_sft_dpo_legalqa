# -*- coding: utf-8 -*-
"""标定 bge-reranker 在法条核验场景的阈值,并验证否定词检测能否补上"否定反转"这个盲区。"""
from pathlib import Path
import sys, re, random, statistics as st, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import legal_check as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = str(Path(__file__).resolve().parents[1])
DB = L.load_law_db(f"{ROOT}/data/stard_corpus.jsonl", verbose=False)
tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/models/bge-reranker-base").eval()

def score(a, b):
    with torch.no_grad():
        x = tok([[a, b]], padding=True, truncation=True, max_length=512, return_tensors="pt")
        return model(**x).logits.view(-1).float().item()

_NEG = re.compile(r"不得|禁止|不予|不能|不应|不属于|不构成|无效|除外|不|未|非|无")
def neg_density(s):
    return len(_NEG.findall(s)) / max(1, len(s))

def flip(s):
    """构造否定反转:给关键判断加否定"""
    for a, b in [("应当", "不应当"), ("可以", "不可以"), ("为夫妻的共同财产", "不是夫妻的共同财产"),
                 ("属于", "不属于"), ("有权", "无权"), ("构成", "不构成"), ("处", "不处")]:
        if a in s:
            return s.replace(a, b, 2)
    return "并非如此。" + s

def paraphrase(s):
    for a, b in [("应当", "应该"), ("可以", "能够"), ("但是", "但"), ("或者", "或"),
                 ("（", "("), ("）", ")"), ("；", ";"), ("的", "")]:
        s = s.replace(a, b)
    return s

rng = random.Random(0)
LAWS = ["刑法", "民法典", "公司法", "劳动合同法"]
samples = []
for law in LAWS:
    keys = [k for k in DB[law] if k[1] == 0 and 40 <= len(DB[law][k]) <= 200]
    for k in rng.sample(keys, min(15, len(keys))):
        samples.append((law, k))

pos, para, neg, flipped = [], [], [], []
pos_ng, para_ng, flip_ng = [], [], []
for law, k in samples:
    truth = DB[law][k]
    claim = truth[:80]
    pos.append(score(truth, claim))
    p = paraphrase(claim); para.append(score(truth, p))
    f = flip(claim);       flipped.append(score(truth, f))
    # 否定词密度差
    pos_ng.append(abs(neg_density(claim) - neg_density(truth)))
    para_ng.append(abs(neg_density(p) - neg_density(truth)))
    flip_ng.append(abs(neg_density(f) - neg_density(truth)))
    # 负例:同一部法律里另取一条的内容
    other = rng.choice([x for x in DB[law] if x != k and len(DB[law][x]) >= 40])
    neg.append(score(truth, DB[law][other][:80]))

def line(name, v):
    v = sorted(v)
    return (f"{name:<12} n={len(v):>3}  最低 {v[0]:>7.2f}  p5 {v[int(len(v)*0.05)]:>7.2f}  "
            f"中位 {st.median(v):>7.2f}  p95 {v[int(len(v)*0.95)]:>7.2f}  最高 {v[-1]:>7.2f}")

print("=== bge-reranker 分数分布 ===")
print(line("正例(逐字)", pos))
print(line("转述例", para))
print(line("否定反转", flipped))
print(line("负例(错配)", neg))

print("\n=== 否定词密度差(|claim - truth|)===")
print(line("正例", pos_ng))
print(line("转述例", para_ng))
print(line("否定反转", flip_ng))

th = min(sorted(para)[int(len(para)*0.05)], sorted(pos)[0])
print(f"\n建议 reranker 阈值:{max(sorted(neg)[-1], 0):.2f} < 阈值 < {th:.2f}")
ng_th = max(sorted(para_ng)[int(len(para_ng)*0.95)], sorted(pos_ng)[-1])
print(f"建议否定词密度差告警线:> {ng_th:.4f}"
      f"(否定反转的中位是 {st.median(flip_ng):.4f})")
