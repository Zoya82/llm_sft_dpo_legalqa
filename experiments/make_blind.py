# -*- coding: utf-8 -*-
"""生成盲评数据:按硬伤组合分层抽样,随机打乱 A/B 顺序并隐去模型名。
映射单独存盘,判完再揭晓 —— 消除位置偏差和"知道哪个是微调模型"的先验偏差。
"""
from pathlib import Path
import json, random, sys, collections

ROOT = str(Path(__file__).resolve().parents[1])
d = json.load(open(f"{ROOT}/outputs/eval/for_judge.json", encoding="utf-8"))
GROUP = sys.argv[1] if len(sys.argv) > 1 else "base_vs_sft"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LEFT, RIGHT = ("base", "sft") if GROUP == "base_vs_sft" else ("sft", "dpo")

# 分层抽样:按"三档合并后的硬伤组合"分层,各层按比例抽,保证难易分布不失真
strata = collections.defaultdict(list)
for i, x in enumerate(d):
    key = tuple(sorted(set(h for m in ["base", "sft", "dpo"] for h in x[f"{m}_核验"]["硬伤"])))
    strata[key].append(i)

rng = random.Random(20260803)
picked = []
for key, idxs in sorted(strata.items(), key=lambda kv: -len(kv[1])):
    k = max(1, round(len(idxs) * N / len(d)))
    picked += rng.sample(idxs, min(k, len(idxs)))
picked = sorted(picked)[:N]

out, mapping = [], {}
for i in picked:
    x = d[i]
    flip = rng.random() < 0.5                       # 随机决定谁当 A
    a_key, b_key = (RIGHT, LEFT) if flip else (LEFT, RIGHT)
    mapping[str(i)] = {"A": a_key, "B": b_key}
    def brief(m):
        v = x[f"{m}_核验"]
        cites = [f"{c['引用']}={c['判定']}" for c in v["法条"]]
        return {"文本": x[m][:320], "字数": v["字数"],
                "硬伤": v["硬伤"], "法条核验": cites}
    out.append({"id": i, "问题": x["q"], "参考答案": x["ref"][:200],
                "答案A": brief(a_key), "答案B": brief(b_key)})

json.dump(out, open(f"/tmp/blind_{GROUP}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(mapping, open(f"/tmp/mapping_{GROUP}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{GROUP}: 抽 {len(out)} 条 / 共 {len(d)} 条(抽样率 {len(out)/len(d):.0%})")
print(f"分层数 {len(strata)},A 位是 {LEFT} 的比例 "
      f"{sum(1 for v in mapping.values() if v['A']==LEFT)/len(mapping):.0%}")
print(f"→ /tmp/blind_{GROUP}.json(评判用) + /tmp/mapping_{GROUP}.json(揭晓用)")
