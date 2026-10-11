# -*- coding: utf-8 -*-
"""检索侧 A/B 实验:① HyDE 改写查询 ② 提高 top-k
只评估检索命中率(不跑最终生成),几分钟出结果。

评判口径:参考答案里引用的法条 = ground truth,看它有没有出现在检索的 top-k 里。
"""
from pathlib import Path
import json, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import legal_check as L
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

ROOT = str(Path(__file__).resolve().parents[1])
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

print("加载法条库与向量…")
corpus = [json.loads(l) for l in open(f"{ROOT}/data/stard_corpus.jsonl", encoding="utf-8") if l.strip()]
vecs = torch.load(f"{ROOT}/data/stard_vecs.pt").to(DEV)
d = json.load(open(f"{ROOT}/outputs/eval/for_judge.json", encoding="utf-8"))

# (法律名,条号,之几) → corpus 下标
idx = {}
for j, c in enumerate(corpus):
    m = L._ART_RE.match(c["name"].strip())
    if m:
        lw, nm = L._norm_law(m.group(1)), L.cn2int(m.group(2))
        sb = L.cn2int(m.group(3)) if m.group(3) else 0
        if nm:
            idx.setdefault((lw, nm, sb), j)

# 只评有 ground truth 的题目
tasks = []
for x in d:
    g = {idx[(l, n, s)] for l, n, s, _ in L.extract_citations(x["ref"]) if (l, n, s) in idx}
    if g:
        tasks.append((x["q"], g))
print(f"可评测题目 {len(tasks)} / {len(d)}\n")

print("加载 bge + reranker…")
etok = AutoTokenizer.from_pretrained(f"{ROOT}/models/bge-small-zh-v1.5")
emb = AutoModel.from_pretrained(f"{ROOT}/models/bge-small-zh-v1.5").to(DEV).eval()
rtok = AutoTokenizer.from_pretrained(f"{ROOT}/models/bge-reranker-base")
rer = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/models/bge-reranker-base").to(DEV).eval()

@torch.no_grad()
def embed(t):
    e = etok([t], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
    return F.normalize(emb(**e).last_hidden_state[:, 0], p=2, dim=1)

@torch.no_grad()
def retrieve_ranked(query, recall_n=100):
    """返回 rerank 后的 corpus 下标序列(长度 recall_n)"""
    sims = (embed(query) @ vecs.T)[0]
    cand = sims.topk(recall_n).indices.tolist()
    pairs = [[query, corpus[i]["content"]] for i in cand]
    sc = []
    for b in range(0, len(pairs), 32):                       # 分批,防 MPS 显存峰值
        e = rtok(pairs[b:b+32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
        sc.append(rer(**e).logits.view(-1))
    sc = torch.cat(sc)
    order = sc.argsort(descending=True).tolist()
    return [cand[j] for j in order]

# ---- HyDE:先让生成模型写一段"法条风格"的假想依据,拿它去检索 ----
print("加载生成模型(HyDE 用)…")
gtok = AutoTokenizer.from_pretrained(f"{ROOT}/outputs/dpo_merged")
gm = AutoModelForCausalLM.from_pretrained(f"{ROOT}/outputs/dpo_merged", dtype=torch.float32).to(DEV).eval()

@torch.no_grad()
def hyde(q, max_new_tokens=96):
    msgs = [{"role": "system", "content": "你是法律专家。"},
            {"role": "user", "content":
             f"针对下面的问题,写一段法条风格的条文文字(不超过60字,不要解释、不要编条号,"
             f"只写法律规定本身的表述):\n{q}"}]
    t = gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    i = gtok(t, return_tensors="pt").to(DEV)
    o = gm.generate(**i, max_new_tokens=max_new_tokens, do_sample=False)
    return gtok.decode(o[0][i["input_ids"].shape[1]:], skip_special_tokens=True).strip()

KS = [3, 5, 8]
res = {f"原问题 k={k}": 0 for k in KS} | {f"HyDE k={k}": 0 for k in KS} | {f"问题+HyDE k={k}": 0 for k in KS}
t0 = time.time()
for n, (q, gold) in enumerate(tasks, 1):
    r_plain = retrieve_ranked(q)
    h = hyde(q)
    r_hyde = retrieve_ranked(h)
    r_both = retrieve_ranked(q + " " + h)          # 拼接:保留原问题信息 + 法条风格
    for k in KS:
        if gold & set(r_plain[:k]): res[f"原问题 k={k}"] += 1
        if gold & set(r_hyde[:k]):  res[f"HyDE k={k}"] += 1
        if gold & set(r_both[:k]):  res[f"问题+HyDE k={k}"] += 1
    if n % 10 == 0:
        el = time.time() - t0
        print(f"  [{n}/{len(tasks)}] {el/60:.1f} 分,预计还需 {el/n*(len(tasks)-n)/60:.0f} 分")

print(f"\n{'='*54}\n检索命中率(top-k 里包含参考答案引用的法条)")
print(f"{'方案':<18}{'k=3':>8}{'k=5':>8}{'k=8':>8}")
print("-" * 44)
for name in ["原问题", "HyDE", "问题+HyDE"]:
    row = "".join(f"{res[f'{name} k={k}']/len(tasks):>8.0%}" for k in KS)
    print(f"{name:<18}{row}")
print(f"\n基线(当前 06 用的是「原问题 k=3」)= {res['原问题 k=3']/len(tasks):.0%}")

# 存一份 HyDE 样例,便于人工看改写质量
samples = [{"问题": q, "HyDE 改写": hyde(q)} for q, _ in tasks[:5]]
json.dump(samples, open("/tmp/hyde_samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("HyDE 改写样例已存 /tmp/hyde_samples.json")
