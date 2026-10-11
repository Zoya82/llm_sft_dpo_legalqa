# -*- coding: utf-8 -*-
"""分层抽样,产出给强模型逐条打分的校准集。

【为什么要这一步】
规则打分器(score_sft_quality.py)的核心项"切题回答"占 40 分,而"这句话有没有
回答问题"本质上需要理解语义 —— 正则做不到。v1→v3 三轮迭代把抽查误判率从 60%+
压下来了,但压不到 0。所以需要一个**金标准**来量化它到底准不准。

【AlpaGasus 的做法 vs 这里的做法】
    AlpaGasus  用 ChatGPT 给全部 52k 条打分(有 API,能全量)
    这里       强模型逐条打 200 条 → 算与规则分的相关性 → 校准规则 → 规则跑全池
后者额外产出一个 AlpaGasus 没有的东西:**规则器与强模型的一致性有多高**,
这个数字本身就是结论 —— 一致性高就信规则(零成本、可复现),低就说明规则不能用。

【抽样设计】
按规则分五等分分层,每层 40 条 —— 不能只抽高分段,否则算不出规则在低分段的表现。

【评分标准(1~5,与 AlpaGasus 同粒度)】
    5  直接回答问题 + 依据充分(法条准确或说理到位) + 无冗余,可直接当标准答案
    4  回答了问题,有小瑕疵(略长 / 依据笼统 / 结论稍晚出现)
    3  基本回答了,但有明显问题(埋得深 / 明显冗余 / 缺依据)
    2  答非所问、和稀泥,或只有边缘信息
    1  完全没回答、或有事实错误

另外单独标一个二元判断 `direct`:**开头是否直接回答了问题**
(对应规则的"切题回答"项,用来算混淆矩阵)。

【用法】
    python3 experiments/sample_for_labeling.py          # 生成 4 批
    python3 experiments/sample_for_labeling.py 1        # 打印第 1 批,供逐条打分
"""
import os, sys, json, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs", "eval")
N_TOTAL, N_LAYER, N_BATCH = 200, 5, 4
SEED = 2026


def build():
    d = json.load(open(os.path.join(OUT, "sft_quality.json"), encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: r["score"])
    n, per = len(rows), N_TOTAL // N_LAYER
    rng = random.Random(SEED)

    picked = []
    for i in range(N_LAYER):                      # 五等分分层,每层等量抽
        layer = rows[i * n // N_LAYER:(i + 1) * n // N_LAYER]
        for r in rng.sample(layer, per):
            picked.append({**r, "layer": i + 1})
    rng.shuffle(picked)                           # 打乱,避免我打分时被分层顺序暗示

    # 需要答案全文 —— sft_quality.json 里没存,从训练池按 idx 取回
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from score_sft_quality import load_full_train_pool
    pool = load_full_train_pool()
    for p in picked:
        p["answer"] = pool[p["idx"]]["answer"]

    size = len(picked) // N_BATCH
    for b in range(N_BATCH):
        batch = picked[b * size:(b + 1) * size] if b < N_BATCH - 1 else picked[b * size:]
        json.dump(batch, open(os.path.join(OUT, f"label_batch_{b+1}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"  label_batch_{b+1}.json  {len(batch)} 条 | "
              f"规则分 {min(x['score'] for x in batch)}~{max(x['score'] for x in batch)}")
    print(f"\n共 {len(picked)} 条,分层各 {per} 条(规则分五等分)")


def show(b):
    batch = json.load(open(os.path.join(OUT, f"label_batch_{b}.json"), encoding="utf-8"))
    print(f"===== 第 {b} 批 · 共 {len(batch)} 条 =====\n")
    for i, x in enumerate(batch, 1):
        print(f"### [{i}] idx={x['idx']} | 规则分 {x['score']} | 类型 {x['问题类型']} | 规则判「{x['回答形态']}」")
        print(f"Q: {x['q']}")
        print(f"A: {x['answer']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(int(sys.argv[1]))
    else:
        build()
