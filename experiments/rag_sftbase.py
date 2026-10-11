# -*- coding: utf-8 -*-
"""RAG 换基座对照:用 sft_merged 而不是 dpo_merged 跑同一套 RAG。

【为什么要跑这个】
06 的 RAG 一直是拿 dpo_merged 当生成模型。而 checkpoint_sweep 实测发现:

    sft_merged   有引用率 59.5% | 引用准确率 35.5%
    dpo_merged   有引用率 45%   | 引用准确率 20%    ← 引用行为已被 DPO 削弱

于是"RAG 有引用率只有 22%""51% 用了检索内容却不标注"这些观察,
被归因成 **0.5B 的 in-context 能力不足 / 训练数据缺少标注行为**。
但那个归因从没排除过一个更简单的解释:**基座模型本身就不爱引法条了**。

唯一变量:生成模型 dpo_merged → sft_merged。检索、prompt(v2)、k=5 全部不变。

    有引用率显著 >22%  → 原归因被推翻,真凶是 DPO 削弱了引用行为
    仍在 22% 上下      → 原归因坐实,确实是模型能力问题
"""
from pathlib import Path
import json, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

ROOT = str(Path(__file__).resolve().parents[1])
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
K = 5

corpus = [json.loads(l) for l in open(f"{ROOT}/data/stard_corpus.jsonl", encoding="utf-8") if l.strip()]
vecs = torch.load(f"{ROOT}/data/stard_vecs.pt").to(DEV)
sft_eval = json.load(open(f"{ROOT}/data/sft_eval.json", encoding="utf-8"))
items = [{"system": m["messages"][0]["content"], "user": m["messages"][1]["content"]} for m in sft_eval]

etok = AutoTokenizer.from_pretrained(f"{ROOT}/models/bge-small-zh-v1.5")
emb = AutoModel.from_pretrained(f"{ROOT}/models/bge-small-zh-v1.5").to(DEV).eval()
rtok = AutoTokenizer.from_pretrained(f"{ROOT}/models/bge-reranker-base")
rer = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/models/bge-reranker-base").to(DEV).eval()
gtok = AutoTokenizer.from_pretrained(f"{ROOT}/outputs/sft_merged")
gm = AutoModelForCausalLM.from_pretrained(f"{ROOT}/outputs/sft_merged", dtype=torch.float32).to(DEV).eval()

SYSTEM = ("你是专业的中文法律咨询助手。请针对用户的法律问题，给出准确、有依据、条理清晰的解答；"
          "涉及具体法条时尽量说明，不要编造不存在的法条或事实。")

@torch.no_grad()
def retrieve(q, k=K, recall_n=100):
    e = etok([q], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
    qv = F.normalize(emb(**e).last_hidden_state[:, 0], p=2, dim=1)
    cand = (qv @ vecs.T)[0].topk(recall_n).indices.tolist()
    pairs = [[q, corpus[i]["content"]] for i in cand]
    sc = []
    for b in range(0, len(pairs), 32):
        x = rtok(pairs[b:b+32], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
        sc.append(rer(**x).logits.view(-1))
    order = torch.cat(sc).argsort(descending=True).tolist()
    return [corpus[cand[j]] for j in order[:k]]

@torch.no_grad()
def gen(user, max_new_tokens=512):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    t = gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    i = gtok(t, return_tensors="pt").to(DEV)
    o = gm.generate(**i, max_new_tokens=max_new_tokens, do_sample=False)
    return gtok.decode(o[0][i["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def user_v2(ctx, q):
    return (f"请参考下面的法条回答问题。要求：\n"
            f"1. 先直接给出结论，再说明理由；\n"
            f"2. 引用法条时注明条号，内容要与所给法条一致；\n"
            f"3. 所给法条没覆盖到的部分，可以结合法律常识补充说明，但不要编造条号；\n"
            f"4. 答案要完整、有条理，不要只写一句话。\n\n"
            f"【参考法条】\n{ctx}\n\n【问题】{q}")

preds, retrieved = [], []
t0 = time.time()
for i, it in enumerate(items, 1):
    hits = retrieve(it["user"])
    ctx = "\n".join(f"{c['name']}：{c['content'].strip()}" for c in hits)
    preds.append(gen(user_v2(ctx, it["user"])))
    retrieved.append([c["name"] for c in hits])
    if i % 20 == 0:
        el = time.time() - t0
        print(f"  [{i}/{len(items)}] {el/60:.1f} 分,预计还需 {el/i*(len(items)-i)/60:.0f} 分", flush=True)

json.dump(preds, open(f"{ROOT}/outputs/eval/preds_rag_sftbase.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(retrieved, open(f"{ROOT}/outputs/eval/rag_retrieved_sftbase.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
import statistics as st
print(f"\n完成,用时 {(time.time()-t0)/60:.1f} 分 | 平均字数 {st.mean([len(p) for p in preds]):.0f}"
      f"(v1 是 109,DPO 是 221)")
