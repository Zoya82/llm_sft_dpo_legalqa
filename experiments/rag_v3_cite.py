# -*- coding: utf-8 -*-
"""RAG prompt v3:强制标注出处 —— 在花 15~20 轮重写训练数据之前,先试零成本的这一招。

【要回答的问题】
RAG 那一档的已知缺陷:**用了检索到的法条内容,却不写出处**。
文档里记的是 51%(3-gram 重合 ≥0.3),本脚本用 0.25 阈值复现是 32% —— 口径待核,
但无论哪个数,绝对量都在 37~59 条之间,不是小问题。

而且它比"格式不好看"严重:实测有一条把「刑事和解协议书」的内容当成「离婚协议书」照抄。
**如果它标注了出处,这个错误当场就会被核验器抓到。** 所以"教标注"的真正价值是
让错误可被检测,不是让答案好看。

【为什么先试 prompt】
根因诊断是"训练数据里从没有'标注引用'这个行为模式,指令遵循弥补不了"。
但这个结论此前是**推断**,没验证过 —— v1/v2 的对照改的是封闭式 vs 开放式,
**从来没有一版 prompt 明确要求过"每个论点都要标注"**。

先跑这一组:
    有效 → 省下 15~20 轮重写,零成本解决
    无效 → "指令遵循弥补不了缺失的行为模式"从推断变成实证,这本身就是结论

【v3 与 v2 的唯一差别】
把"引用法条时注明条号"(软要求)改成"每个论点后必须标注"(硬要求)+ **给出格式示例**。
few-shot 示例是关键 —— 0.5B 模型对纯指令的遵循能力弱,给样子比讲道理有效。

【用法】
    python3 experiments/rag_v3_cite.py
产物:outputs/eval/preds_rag_v3.json(不覆盖 v2 的产物,保留对照)
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
gtok = AutoTokenizer.from_pretrained(f"{ROOT}/outputs/dpo_merged")
gm = AutoModelForCausalLM.from_pretrained(f"{ROOT}/outputs/dpo_merged", dtype=torch.float32).to(DEV).eval()

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


def user_v3(ctx, q):
    """v3:硬性要求逐点标注 + 给格式示例。与 v2 的差别只在第 2 条和示例段。"""
    return (
        f"请参考下面的法条回答问题。要求：\n"
        f"1. 先直接给出结论，再说明理由；\n"
        f"2. **每一个论点后面都必须标注依据的法条**，格式为（《法律名》第N条）；\n"
        f"3. 只能标注【参考法条】中出现的条号，不得编造；找不到依据的论点写（无对应法条）；\n"
        f"4. 答案要完整、有条理，不要只写一句话。\n\n"
        f"【格式示例】\n"
        f"夫妻共同财产原则上均等分割（《民法典》第1087条）。"
        f"但一方有隐藏、转移财产行为的，可以少分或者不分（《民法典》第1092条）。\n\n"
        f"【参考法条】\n{ctx}\n\n【问题】{q}")


if __name__ == "__main__":
    preds, retrieved = [], []
    t0 = time.time()
    print(f"RAG v3(强制标注)全量生成 {len(items)} 条,k={K}", flush=True)
    for i, it in enumerate(items, 1):
        hits = retrieve(it["user"])
        ctx = "\n".join(f"{c['name']}：{c['content'].strip()}" for c in hits)
        preds.append(gen(user_v3(ctx, it["user"])))
        retrieved.append([c["name"] for c in hits])
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(items)}] {el/60:.1f} 分,预计还需 {el/i*(len(items)-i)/60:.0f} 分", flush=True)

    json.dump(preds, open(f"{ROOT}/outputs/eval/preds_rag_v3.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(retrieved, open(f"{ROOT}/outputs/eval/rag_retrieved_v3.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 与 v2 直接对比 ----
    import re, statistics as st
    import legal_check as L
    db = L.load_law_db(f"{ROOT}/data/stard_corpus.jsonl", verbose=False)
    v2 = json.load(open(f"{ROOT}/outputs/eval/preds_rag.json", encoding="utf-8"))
    ret2 = json.load(open(f"{ROOT}/outputs/eval/rag_retrieved.json", encoding="utf-8"))

    def title2content(t):
        m = re.match(r'(.+?)第([一二三四五六七八九十百千零〇\d]+)条(之[一二三四五六七八九])?$', t)
        if not m:
            return None
        law = L._norm_law(m.group(1)); num = L.cn2int(m.group(2))
        sub = L.cn2int(m.group(3)[1:]) if m.group(3) else 0
        arts = db.get(law)
        return arts.get((num, sub)) if arts else None

    def report(name, ps, rs):
        ok = bad = withc = nocite_used = nocite = 0
        for a, ts in zip(ps, rs):
            cs = L.verify_citations(a, db)
            o = sum(1 for c in cs if c["verdict"] == "ok")
            b = sum(1 for c in cs if c["verdict"] in ("misattributed", "not_found", "out_of_range"))
            ok += o; bad += b
            if L.extract_citations(a):
                withc += 1
            else:
                nocite += 1
                for t in ts:
                    c = title2content(t)
                    # 方向很要紧:要问的是"回答里有多少内容来自条文",不是反过来。
                    # 回答平均 101 字、条文动辄几百字,用 (条文→回答) 会被条文长度稀释:
                    # 同一批数据,(条文→回答)算出 28%,(回答→条文)算出 44%。后者才对。
                    if c and L._char_overlap(a, c) >= 0.30:
                        nocite_used += 1; break
        print(f"  {name:<10}{st.mean(len(p) for p in ps):>7.0f}字{withc/len(ps):>9.0%}"
              f"{ok/max(1,ok+bad):>10.0%}{ok:>7}{bad:>6}"
              f"{nocite_used:>10} ({nocite_used/max(1,nocite):.0%} of 无标注)")

    print(f"\n{'':<10}{'平均字数':>8}{'有标注率':>9}{'引用准确率':>10}{'对':>6}{'错':>6}{'用了没标注':>12}")
    print("  " + "-" * 66)
    report("v2 开放式", v2, ret2)
    report("v3 强制标注", preds, retrieved)
    print("\n判读:")
    print("  有标注率↑且引用准确率不崩 → prompt 能解决,不必重写训练数据")
    print("  有标注率纹丝不动          → 指令遵循确实弥补不了缺失的行为模式,需要 RAFT")
    print("  ⚠️ 引用准确率**下降是预期内的** —— 原来隐藏在'没标注'里的错误暴露出来了,")
    print("     这是测量变准,不是模型变差。要看的是'对'的绝对条数有没有增加。")
    print(f"\n完成,用时 {(time.time()-t0)/60:.1f} 分")
