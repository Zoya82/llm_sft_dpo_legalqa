# -*- coding: utf-8 -*-
"""规模化标定 NLI 在法条核验上的表现,并和 reranker 做同一批样本的对照。"""
from pathlib import Path
import sys, re, random, statistics as st, torch, collections
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import legal_check as L
from transformers import BertTokenizer, BertForSequenceClassification

ROOT = str(Path(__file__).resolve().parents[1])
DB = L.load_law_db(f"{ROOT}/data/stard_corpus.jsonl", verbose=False)
M = f"{ROOT}/models/Erlangshen-Roberta-110M-NLI"
tok = BertTokenizer.from_pretrained(M)
model = BertForSequenceClassification.from_pretrained(M).eval()
ID2 = model.config.id2label

def nli(premise, hypothesis):
    with torch.no_grad():
        x = tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        p = torch.softmax(model(**x).logits[0], -1)
    i = int(p.argmax())
    return ID2[i], float(p[i])

def paraphrase_light(s):
    for a, b in [("应当", "应该"), ("可以", "能够"), ("但是", "但"), ("或者", "或"), ("的", "")]:
        s = s.replace(a, b)
    return s

def flip(s):
    for a, b in [("应当", "不应当"), ("可以", "不可以"), ("属于", "不属于"), ("有权", "无权"),
                 ("构成", "不构成"), ("为夫妻", "不是夫妻"), ("处", "不处"), ("承担", "不承担")]:
        if a in s:
            return s.replace(a, b, 2)
    return "以下说法不成立:" + s

rng = random.Random(0)
LAWS = ["刑法", "民法典", "公司法", "劳动合同法", "道路交通安全法"]
samples = []
for law in LAWS:
    keys = [k for k in DB[law] if k[1] == 0 and 40 <= len(DB[law][k]) <= 200]
    for k in rng.sample(keys, min(20, len(keys))):
        samples.append((law, k))

# 四类样本 → 期望标签
cases = {"正例(逐字)": "ENTAILMENT", "转述例": "ENTAILMENT",
         "否定反转": "CONTRADICTION", "负例(错配)": "NEUTRAL"}
res = {k: collections.Counter() for k in cases}
conf = {k: [] for k in cases}

for law, k in samples:
    truth = DB[law][k]
    pairs = {
        "正例(逐字)": truth[:80],
        "转述例":    paraphrase_light(truth[:80]),
        "否定反转":   flip(truth[:80]),
        "负例(错配)": DB[law][rng.choice([x for x in DB[law] if x != k and len(DB[law][x]) >= 40])][:80],
    }
    for name, claim in pairs.items():
        lab, p = nli(truth, claim)
        res[name][lab] += 1
        conf[name].append(p)

print(f"样本:{len(samples)} 条真实条文 × 4 类 = {len(samples)*4} 次判断\n")
print(f"{'类型':<12}{'期望标签':<16}{'ENTAIL':>8}{'CONTRA':>8}{'NEUTRAL':>9}{'命中率':>9}{'平均置信':>9}")
print("-" * 74)
for name, want in cases.items():
    c = res[name]; n = sum(c.values())
    print(f"{name:<12}{want:<16}{c['ENTAILMENT']:>8}{c['CONTRADICTION']:>8}{c['NEUTRAL']:>9}"
          f"{c[want]/n:>9.0%}{st.mean(conf[name]):>9.3f}")

# 关键指标:法律场景最怕"说反了却判成对"
flip_as_ok = res["否定反转"]["ENTAILMENT"]
neg_as_ok  = res["负例(错配)"]["ENTAILMENT"]
para_as_bad = res["转述例"]["CONTRADICTION"] + res["转述例"]["NEUTRAL"]
print(f"\n关键错误:")
print(f"  否定反转被判成 ENTAILMENT(说反了却算对):{flip_as_ok}/{len(samples)}")
print(f"  错配负例被判成 ENTAILMENT(张冠李戴却算对):{neg_as_ok}/{len(samples)}")
print(f"  转述例被判成 非ENTAILMENT(对的被误杀):{para_as_bad}/{len(samples)}")

# 置信度门槛扫描:只在高置信时才采信 NLI
print(f"\n置信度门槛的影响(仅统计 ENTAILMENT 判定):")
for th in [0.0, 0.5, 0.7, 0.9]:
    ok_para = sum(1 for (law,k) in samples
                  if (lambda r: r[0]=="ENTAILMENT" and r[1]>=th)(nli(DB[law][k], paraphrase_light(DB[law][k][:80]))))
    print(f"  门槛 {th}: 转述例中被采信为 ENTAILMENT 的 {ok_para}/{len(samples)}")
