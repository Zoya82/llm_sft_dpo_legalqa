# -*- coding: utf-8 -*-
"""SFT 训练数据质量打分器 v3 —— 为 AlpaGasus 式筛选提供依据。

【v1 的四个缺陷,都是被数据打脸打出来的】
    ① 误杀:"1、工伤发生后，劳动者**可以**在伤情稳定后申请鉴定" —— 结论就在第一句,
       只因为以序号开头就判 0 分。**序号开头 ≠ 不给结论。**
    ② 误放(方向性错误):"不一定，需要看具体情况来定" 拿了满分 ——
       v1 用"是/不是/可以"这类判断词检测结论,"是否""不一定"正好命中。
       讽刺的是 02 的 corrupt_vague 造的负样本就是这种和稀泥开头,
       **一边当负样本,一边给满分**。
    ③ 零区分度:"切题"98% 满分、"无小节残留" 100% 满分(02 清洗时已处理干净),
       这 25 分白给,实际只在 75 分区间里排序。
    ④ 分数挤成一坨:63% 的样本落在 6 个离散值上,切分点附近大量并列。

【v2 的六项】
    ① 首句质量 30   分四档,**把"和稀泥"单独摘出来判 0**;序号开头先剥掉再判内容
    ② 问答匹配 15   新增:是否类问题要有是/否,数量类要有数字,方式类要有步骤
    ③ 法条引用 20   改进:区分"问题需不需要法条",没引用不再一刀切
    ④ 无冗余   15   替换废掉的"无小节残留":查冗余尾巴 + 长度连续惩罚
    ⑤ 不复读   10   连续化
    ⑥ 信息密度 10   替换废掉的"切题":实质信息(数字/期限/条件/法条)的密度

【定位】
本打分器只用于**排序筛选**,不是绝对质量评价。它的准确率需要用强模型标注的
金标准来校准 —— 见 experiments/sample_for_labeling.py。

【用法】
    python3 experiments/score_sft_quality.py
产物:outputs/eval/sft_quality.json
"""
import os, sys, json, re, statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import legal_check as L

DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")

# ---------- 正则表 ----------
_LIST_HEAD = re.compile(r"^\s*(?:[1-9１-９][、.\.)）]|[一二三四五六七八九][、.]|\([1-9]\)|（[1-9]）|[-*•])\s*")

# 和稀泥 —— v1 最大的漏洞:这些都含"判断词"但等于没回答,一票判 0
_VAGUE = re.compile(r"不一定|不确定|要看|需要看|得看|视具体|视情况|具体情况具体|具体分析|"
                    r"因人而异|因案而异|要根据具体|由.{0,8}决定|没有统一|不能一概而论|"
                    r"要结合具体|应结合具体|建议咨询|建议委托|请咨询|咨询专业律师")

# 【v3 核心改动】不同问题类型,"回答了"长得不一样 —— v2 用同一套标准套所有类型,
# 把"包括:1、2、3"(对"有哪些"的标准答法)和"建议向劳动监察部门投诉"(对"怎么办"的
# 标准答法)全判成了"铺垫",抽查 16 条误判 10 条。
_ANSWER_BY_TYPE = {
    "是否类": re.compile(r"不是|不可以|不能|不属于|不需要|无需|不构成|无效|不会|不得|不违法|不合法|"
                      r"没有|无权|可以|能够|属于|需要|构成|有效|会被|会判|违法|合法|有权|有.{0,8}权利|"
                      r"应当|必须|算|是的|要"),
    "数量类": re.compile(r"[0-9０-９]+\s*(?:年|个月|月|日|天|周|小时|倍|%|％|元|万元|万|分)|"
                      r"[一二三四五六七八九十百千两]+(?:年|个月|日|天|周|倍|元|万元|分)|"
                      r"以上|以下|不超过|不低于|至少|最高|最低"),
    "方式类": re.compile(r"建议|可以向|应当向|需要向|可以到|应(当)?(先|向|到|提交|准备|办理|申请)|"
                      r"提交|申请|办理|签订|准备|携带|起诉|报案|投诉|协商|仲裁|"
                      r"第一步|首先|流程如下|程序如下|步骤|方式(如下|有)|途径"),
    "定义类": re.compile(r"是指|指的是|是一种|是一个|通常是指|所谓|即|是.{2,20}的(行为|制度|权利|责任|方式)"),
    "列举类": re.compile(r"包括|分为|有以下|如下|有下列|主要有|下列|以下几(种|类|点|方面)|"
                      r"[1-9１-９一二三四五六七八九][、.]"),
}
# 判不出类型时的通用判断词
_VERDICT_GENERIC = re.compile(r"不是|不可以|不能|不属于|无权|不需要|无需|不构成|无效|不予|不会|不得|"
                              r"可以|能够|属于|归|应当|必须|需要|有权|构成|有效|会被|将被|判处|承担|"
                              r"包括|分为|建议|如下|违法|合法")

_TAIL = re.compile(r"(以上(就)?是|综上所述).{0,20}(内容|介绍|解答|回答)|"
                   r"如(果)?(您|你)?还有.{0,8}(疑问|问题)|"
                   r"欢迎.{0,6}咨询|更多.{0,6}(法律)?(问题|资讯)|"
                   r"本文.{0,10}(仅供参考|由.{0,8}整理)")

_HARD_INFO = re.compile(r"[0-9一二三四五六七八九十百千万]+(?:年|月|日|天|周|个月|倍|%|元|万元|分)|"
                        r"第[一二三四五六七八九十百千零〇\d]{1,10}条|"
                        r"以上|以下|不超过|不低于|自.{0,10}起|届满|之日起")

# 问题类型识别(顺序有讲究:先判最具体的)
def question_type(q):
    if re.search(r"多少|几年|几个月|几天|多久|多长时间|期限|比例|标准是|多少钱", q):  return "数量类"
    if re.search(r"有哪些|包括哪些|哪些|分为几|有几种|列举", q):                    return "列举类"
    if re.search(r"(吗|么)[?？]?$|是否|能不能|可不可以|算不算|会不会|要不要|用不用", q): return "是否类"
    if re.search(r"怎么|怎样|如何|该(怎|如)|办理|流程|步骤|需要准备|注意什么", q):     return "方式类"
    if re.search(r"^什么(是|叫)|是什么|指什么|的定义|什么意思", q):                  return "定义类"
    return "其他"


def first_sentence(t):
    m = re.split(r"(?<=[。；;！？])", t.strip())
    return (m[0] if m else t)[:150]


def score_one(q, a, db):
    """返回 (总分 0~100, 各项明细, 诊断信息)。只用于排序,不是绝对质量评价。"""
    s = a.strip()
    fs_raw = first_sentence(s)
    d, meta = {}, {}
    qt = question_type(q)
    meta["问题类型"] = qt

    # ---------- ① 切题回答 40 ----------
    # v2 把"首句质量"和"问答匹配"拆成两项,判定重叠且都用同一套词表;
    # v3 合并成一项,**按问题类型选对应的判定规则**,窗口从 40 字放宽到 120 字
    # (中文法律表述常把主句放在一串从句之后)。
    head = re.sub(r"\s", "", s)[:120]
    if _VAGUE.search(head[:60]):
        d["切题回答"], meta["回答形态"] = 0, "和稀泥"
    else:
        pat = _ANSWER_BY_TYPE.get(qt, _VERDICT_GENERIC)
        listy = bool(_LIST_HEAD.match(fs_raw))
        if pat.search(head):
            d["切题回答"] = 34 if listy else 40      # 序号开头:内容对但不够开门见山
            meta["回答形态"] = "直接回答(序号开头)" if listy else "直接回答"
        elif pat.search(re.sub(r"\s", "", s)[:300]):
            d["切题回答"], meta["回答形态"] = 22, "回答了但埋得深"
        else:
            d["切题回答"], meta["回答形态"] = 10, "没答到点上"

    # ---------- ② 法条引用 20 ----------
    need_law = bool(re.search(r"罪|刑|判|违法|合法|有效|无效|责任|赔偿|权利|义务|时效|标准", q))
    cites = L.verify_citations(s, db)
    ok  = sum(1 for c in cites if c["verdict"] == "ok")
    bad = sum(1 for c in cites if c["verdict"] in ("misattributed", "not_found", "out_of_range"))
    d["法条引用"] = 0 if bad else (20 if ok else (8 if need_law else 15))
    meta.update({"引用对": ok, "引用错": bad, "需要法条": need_law})

    # ---------- ③ 无冗余 15 ----------
    n = len(s)
    has_tail = bool(_TAIL.search(s))
    length_pt = 10 if n <= 350 else (8 if n <= 500 else (4 if n <= 700 else 0))
    d["无冗余"] = length_pt + (0 if has_tail else 5)
    meta.update({"字数": n, "冗余尾巴": has_tail})

    # ---------- ④ 不复读 10 ----------
    rep = L.repetition_rate(s)
    d["不复读"] = round(max(0.0, min(1.0, (0.20 - rep) / 0.20)) * 10)
    meta["复读率"] = round(rep, 3)

    # ---------- ⑤ 信息密度 15 ----------
    # v2 拍了个"3 条/百字"的满分线,实测那是 P90,只有 9% 能拿满分 → 这一项几乎不产生区分度。
    # v3 用实测 P75(约 1.3)做满分线,让分数真正散开。
    hard = len(_HARD_INFO.findall(s))
    dens = hard / max(1, n / 100)
    d["信息密度"] = round(max(0.0, min(1.0, dens / 1.3)) * 15)
    meta["硬信息/百字"] = round(dens, 2)

    return sum(d.values()), d, {"首句": fs_raw, **meta}


FULL = {"切题回答": 40, "法条引用": 20, "无冗余": 15, "不复读": 10, "信息密度": 15}


def load_full_train_pool():
    """复刻 02 的划分+清洗,拿到**全部** 5353 条训练侧数据。

    ⚠️ 关键:给**全池**打分,不是给已经截取的前 2000 条打分。
    从 5353 条里筛 top-2000,数据量不变、质量提升 —— 筛选就不再是减法了。
    (AlpaGasus 从 52k 筛 9k 是减法,因为它没有更大的池子;这里有。)
    """
    import random
    random.seed(42)
    pool = json.load(open(os.path.join(DATA, "qa_pool.json"), encoding="utf-8"))
    random.shuffle(pool)
    train_pool = pool[150:]                       # 前 150 条留作评测,严格隔离

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
            qq = title if re.search(r'吗|呢|怎么|如何|什么|哪些|多少|是否|能否', title) else title + '是什么'
            qq = qq.rstrip('?？') + '?'
            if usable(qq, body):
                clean.append({'question': qq, 'answer': body})
    return clean


if __name__ == "__main__":
    full = load_full_train_pool()
    N_SFT = 2000
    db = L.load_law_db(os.path.join(DATA, "stard_corpus.jsonl"), verbose=False)
    print(f"训练池 {len(full)} 条 | 当前 02 取的是**前 {N_SFT} 条**(shuffle 顺序,没按质量挑过)")
    print(f"用 v2 打分器给全部 {len(full)} 条打分...\n", flush=True)

    rows = []
    for i, x in enumerate(full):
        total, detail, meta = score_one(x["question"], x["answer"], db)
        rows.append({"idx": i, "q": x["question"], "score": total,
                     "detail": detail, "in_current_2000": i < N_SFT, **meta})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(full)}", flush=True)

    scores = [r["score"] for r in rows]
    print("\n" + "=" * 70)
    print(f"v2 质量分布(满分 100,中位 {st.median(scores):.0f},均值 {st.mean(scores):.1f})")
    print("=" * 70)
    for lo, hi in [(0,30),(30,45),(45,55),(55,65),(65,75),(75,85),(85,101)]:
        n = sum(1 for s in scores if lo <= s < hi)
        print(f"  {lo:>3}~{hi-1:<3} {n:>5} ({n/len(scores):>4.0%}) {'█' * round(n/len(scores)*50)}")
    uniq = len(set(scores))
    top6 = Counter(scores).most_common(6)
    print(f"\n  区分度: {uniq} 个不同分值 | 最常见 6 个分值覆盖 "
          f"{sum(v for _, v in top6)/len(rows):.0%}   (v1 是 63%,越低越好)")

    print("\n回答形态分布(v3 按问题类型分别判定):")
    for k, v in Counter(r["回答形态"] for r in rows).most_common():
        print(f"  {k:<16}{v:>5} ({v/len(rows):>4.0%})")

    print("\n单项均分(满分):")
    for k, f in FULL.items():
        print(f"  {k:<10}{st.mean(r['detail'][k] for r in rows):>6.1f} / {f}")

    print("\n不同阈值下剩多少:")
    for th in [55, 60, 65, 70, 75, 80, 85, 90]:
        n = sum(1 for s in scores if s >= th)
        print(f"  >= {th:<3} → {n:>5} ({n/len(rows):>4.0%})")

    cur = [r for r in rows if r["in_current_2000"]]
    top = sorted(rows, key=lambda r: -r["score"])[:N_SFT]
    print("\n" + "=" * 70)
    print(f"同样取 {N_SFT} 条:当前取法 vs 按 v2 质量取")
    print("=" * 70)
    print(f"  {'':<20}{'当前(前2000)':>14}{'top2000':>12}")
    print(f"  {'平均分':<20}{st.mean(r['score'] for r in cur):>14.1f}{st.mean(r['score'] for r in top):>12.1f}")
    for k in FULL:
        print(f"  {k:<20}{st.mean(r['detail'][k] for r in cur):>14.1f}{st.mean(r['detail'][k] for r in top):>12.1f}")
    for nm, fn in [("和稀泥(坏)", lambda r: r["回答形态"] == "和稀泥"),
                   ("没答到点上(坏)", lambda r: r["回答形态"] == "没答到点上"),
                   ("引用有错(坏)",  lambda r: r["引用错"] > 0),
                   ("冗余尾巴(坏)",  lambda r: r["冗余尾巴"]),
                   ("超500字(坏)",   lambda r: r["字数"] > 500)]:
        print(f"  {nm:<20}{sum(1 for r in cur if fn(r))/len(cur):>13.0%}"
              f"{sum(1 for r in top if fn(r))/len(top):>12.0%}")
    print(f"\n  top2000 最低分 {min(r['score'] for r in top)} | "
          f"与当前 2000 条重合 {len(set(r['idx'] for r in cur) & set(r['idx'] for r in top))/N_SFT:.0%}")

    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    json.dump({"版本": "v3", "满分构成": FULL, "n": len(rows),
               "中位": st.median(scores), "均值": round(st.mean(scores), 1),
               "rows": rows},
              open(os.path.join(OUT, "eval", "sft_quality.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已存 outputs/eval/sft_quality.json")
