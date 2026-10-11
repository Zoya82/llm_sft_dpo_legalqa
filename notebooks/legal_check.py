# -*- coding: utf-8 -*-
"""法条引用核验 + 生成质量客观指标(可离线、可复现、无需调用大模型)"""
import os, re, json
from collections import Counter

# ============ 1. 中文数字 → 阿拉伯数字 ============
_CN = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
_UNIT = {"十":10, "百":100, "千":1000}

def cn2int(s):
    """'三百九十二'->392  '一千零四十二'->1042  '十九'->19  '258'->258"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN:
            number = _CN[ch]
        elif ch in _UNIT:
            unit = _UNIT[ch]
            section += (number if number else 1) * unit   # "十九"的十前面没数字,补1
            number = 0
        else:
            return None
    total = section + number
    return total or None


# ============ 2. 加载法条库,建索引 ============
def _norm_law(name):
    """法律名归一化:去书名号/空格/'中华人民共和国'前缀,便于简称匹配"""
    n = name.replace("《", "").replace("》", "").replace(" ", "").strip()
    n = re.sub(r"^中华人民共和国", "", n)
    return n

_ART_RE = re.compile(r"^(.*?)第([一二三四五六七八九十百千零〇\d]+)条(?:之([一二三四五六七八九十]+))?$")

def load_law_db(path, verbose=True):
    """返回 {归一化法律名: {(条号int, 之几int): 条文text}}

    键必须带"之几":刑法里「第234条」(故意伤害罪)和「第234条之一」(组织出卖人体器官)
    是两个完全不同的条文,修正案新增的"之一/之二"很多。早期版本把"之一"吞掉,
    导致两者互相覆盖,把大量正确引用误判成张冠李戴。

    重复键的处理:后写入的覆盖先写入的,但**内容不一致时显式告警**。
    法律语料常因修订收录多版本,静默丢弃任何一方都可能让核验拿错基准 ——
    与其猜"保留哪个对",不如把冲突暴露出来,由使用者决定。
    (实测这份 stard_corpus:55348 个键、零重复,该分支不会触发。)
    """
    db, conflicts = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            m = _ART_RE.match(d["name"].strip())
            if not m:
                continue
            law, num = _norm_law(m.group(1)), cn2int(m.group(2))
            sub = cn2int(m.group(3)) if m.group(3) else 0
            if not law or num is None:
                continue
            content = d["content"].replace("\\n", "").strip()
            arts = db.setdefault(law, {})
            key = (num, sub)
            if key in arts and arts[key] != content:
                conflicts.append((law, key, arts[key], content))
            arts[key] = content

    if conflicts and verbose:
        print(f"⚠️  法条库有 {len(conflicts)} 处同一条号、内容不一致的重复项(已取最后出现的版本):")
        for law, (num, sub), old, new in conflicts[:5]:
            label = f"第{num}条" + (f"之{sub}" if sub else "")
            print(f"   {law}{label}")
            print(f"     先:{old[:50]}")
            print(f"     后:{new[:50]}")
        if len(conflicts) > 5:
            print(f"   …另有 {len(conflicts)-5} 处")
    return db


# ============ 3. 从回答里抽取法条引用 ============
# 形如:《中华人民共和国刑法》第三百一十条 / 《民法典》第1042条 / 刑法第258条
_CITE_RE = re.compile(
    r"(?:《\s*(?P<l1>[^》]{2,30}?)\s*》|(?P<l2>[一-龥]{2,20}?(?:法典|法|条例|规定|解释|办法)))"
    r"\s*第\s*(?P<num>[一二三四五六七八九十百千零〇\d]{1,10})\s*条"
    r"(?:\s*之\s*(?P<sub>[一二三四五六七八九十]{1,3}))?"
)

# 已被《民法典》(2021.1.1 施行)废止的单行法——引用这些属于"时效性错误"
REPEALED = {"合同法": "民法典合同编", "物权法": "民法典物权编", "担保法": "民法典担保制度",
            "侵权责任法": "民法典侵权责任编", "婚姻法": "民法典婚姻家庭编",
            "继承法": "民法典继承编", "收养法": "民法典婚姻家庭编",
            "民法通则": "民法典总则编", "民法总则": "民法典总则编"}

# 引用前常见的虚词/指代,需从法律名左侧剥掉;剥完是指代词的直接丢弃
_LEAD = re.compile(r"^(?:但是|但|因此|所以|同时|并且|而且|根据|依据|按照|参照|其中|另外|以及|和|与|及|该|本|此)+")
_PRONOUN = {"本法", "该法", "此法", "上法", "前法", "法"}

def extract_citations(text):
    """返回 [(法律名, 条号int, 之几int, 在原文中的结束位置)]"""
    out = []
    for m in _CITE_RE.finditer(text):
        law = _norm_law(m.group("l1") or m.group("l2") or "")
        law = _LEAD.sub("", law)                 # "但是本法" -> "法";"根据刑法" -> "刑法"
        num = cn2int(m.group("num"))
        sub = cn2int(m.group("sub")) if m.group("sub") else 0
        if law and num and law not in _PRONOUN:  # 指代上文的"本法第X条"无法独立核验,跳过
            out.append((law, num, sub, m.end()))
    return out


# ============ 4. 核验单条引用 ============
def _char_overlap(a, b, n=3):
    """3-gram 覆盖:模型宣称的内容里有多少 3 字片段真的出现在该条文里(0~1)。
    为什么是 3 而不是 2 或 1:中文法律文本套话极多——"处三年以下有期徒刑"这种
    量刑表述在几百个条文里通用,单字/双字重合会把张冠李戴的引用误判为正确
    (实测:2-gram 下"刑法310条+重婚量刑"这条幻觉被误判为 ok,3-gram 后正确识别)。"""
    if len(a) < n or len(b) < n:
        return 0.0
    ga = [a[i:i+n] for i in range(len(a) - n + 1)]
    cb = Counter(b[i:i+n] for i in range(len(b) - n + 1))
    hit = 0
    for g in ga:
        if cb.get(g, 0) > 0:
            cb[g] -= 1
            hit += 1
    return hit / len(ga)

def verify_citations(text, db, window=60, hi=0.50, lo=0.25):
    """
    对回答里每条法条引用给出判定:
      ok            条号存在,且后文描述与真实条文内容对得上(sim>=hi)
      uncertain     介于两者之间——多半是转述而非原文照抄,交给人工/LLM 复核
      misattributed 条号存在,但后文描述与该条内容严重不符(张冠李戴 → 幻觉, sim<lo)
      out_of_range  条号超出该法最大条号(确凿编造)
      not_found     该法在库中,但没有这一条(疑似编造,受法条库完整度影响)
      unverifiable  该法不在库中,无法核验
    阈值用法条库自身标定:逐字引用正例 sim 中位 1.00、最小 0.96;
    条号内容错配负例 sim 中位 0.09、最大 0.46。
    """
    res, seen = [], set()
    full = re.sub(r"\s", "", text)
    for law, num, sub, end in extract_citations(text):
        key = (num, sub)
        label = f"第{num}条" + (f"之{sub}" if sub else "")
        if (law, key) in seen:   # 同一条法条在一篇回答里反复提及,只判一次
            continue             # (论述中的二次提及往往不复述条文,重复判会造成假阳性)
        seen.add((law, key))
        base = {"law": law, "num": num, "sub": sub, "label": label}
        if law in REPEALED:      # 时效性错误:该法已废止,应改引民法典
            res.append({**base, "verdict": "repealed",
                        "sim": None, "should_be": REPEALED[law]})
            continue
        arts = db.get(law)
        if arts is None:
            res.append({**base, "verdict": "unverifiable", "sim": None})
            continue
        if key in arts:
            claim = re.sub(r"\s", "", text[end:end + window])
            claim = re.sub(r"^[的规定内容为是：:，,、“\"']+", "", claim)
            # 两个方向都看,取更宽的那个:
            #  ① 引用后紧跟的文字是不是在复述该条文(抓"张冠李戴")
            #  ② 该条文的内容有没有出现在回答的任何位置(避免把"论述中提及"误判为编造)
            sim = max(_char_overlap(claim, arts[key]), _char_overlap(arts[key], full))
            if len(claim) < 15:          # 只报条号、几乎没说内容 → 证据不足,不判对错
                verdict = "uncertain"
            else:
                verdict = "ok" if sim >= hi else ("misattributed" if sim < lo else "uncertain")
            res.append({**base, "verdict": verdict, "sim": round(sim, 3),
                        "truth": arts[key][:60],
                        "_claim": claim, "_truth_full": arts[key]})  # 供语义复核用
        elif num > max(n for n, _ in arts):
            res.append({**base, "verdict": "out_of_range", "sim": None,
                        "max": max(n for n, _ in arts)})
        else:
            res.append({**base, "verdict": "not_found", "sim": None})
    return res


# ============ 4.5 语义复核(可选):把 uncertain 那一档交给语义模型 ============
# 字面比对(3-gram)判不了两种情况:
#   ① 同义改写:"应当承担赔偿责任" vs "需要负责赔偿" —— 字面零重合,意思一样
#   ② 否定反转:"不承担责任" vs "承担责任"         —— 字面高度重合,意思相反
#
# 【当前方案】NLI(自然语言推理)+ 否定词密度告警 —— 见下方 semantic_recheck
#
# 【已评估并否决:bge-reranker】保留函数供复现对比,**不在主流程中调用**。
# 交叉编码器建模的是"相关性"(relevance)而非"一致性",实测:
#     完全无关   -5.40   ← 只有这一档判得准
#     深度转述    4.92   ← 比否定反转还低,排序是反的
#     否定反转    5.50   ← 与正例区间重叠,0% 识别率
# 早期标定曾给出 7.0 这个"安全阈值",是因为标定用的转述样本太温和(只换同义词),
# 不代表模型真实的改写幅度 —— 这是本项目第二次栽在"合成样本不代表真实分布"上。
_RERANKER = None

_NEG = re.compile(r"不得|禁止|不予|不能|不应|不属于|不构成|无效|除外|不|未|非|无")

def _neg_density(s):
    """否定词密度:否定标记数 / 字符数。两段文本密度差异大 → 可能有否定反转"""
    return len(_NEG.findall(s)) / max(1, len(s))

def load_reranker(path):
    """[已否决方案,保留供对比复现] 懒加载交叉编码器。主流程用 NLI,不走这里"""
    global _RERANKER
    if _RERANKER is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        _RERANKER = (tok, model, torch)
    return _RERANKER

def unload_reranker():
    """用完释放(模型约 1.1GB,MPS 上内存紧张时记得调)"""
    global _RERANKER
    _RERANKER = None
    import gc
    gc.collect()

def rerank_score(truth, claim, model_path):
    """交叉编码器打分:两段文本的语义相关性,分数越高越相关(logit,约 -10 ~ +10)"""
    tok, model, torch = load_reranker(model_path)
    with torch.no_grad():
        x = tok([[truth, claim]], padding=True, truncation=True,
                max_length=512, return_tensors="pt")
        return model(**x).logits.view(-1).float().item()

# ---- NLI:自然语言推理,判断真实条文与模型描述是「蕴含 / 矛盾 / 中立」----
_NLI = None

def load_nli(path):
    """懒加载 NLI 模型(Erlangshen-Roberta-110M-NLI,约 400MB)"""
    global _NLI
    if _NLI is None:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        tok = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path).eval()
        _NLI = (tok, model, torch)
    return _NLI

def unload_nli():
    global _NLI
    _NLI = None
    import gc
    gc.collect()

def nli_judge(premise, hypothesis, model_path):
    """返回 (标签, 置信度)。标签为 ENTAILMENT / CONTRADICTION / NEUTRAL
    premise = 法条库里的真实条文,hypothesis = 模型宣称这条法条说了什么"""
    tok, model, torch = load_nli(model_path)
    with torch.no_grad():
        x = tok(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt")
        p = torch.softmax(model(**x).logits[0], -1)
    i = int(p.argmax())
    return model.config.id2label[i], float(p[i])


def semantic_recheck(cites, nli_path, neg_th=0.013, min_conf=0.5):
    """对 verdict == 'uncertain' 的引用做语义复核,原地改写判定。

    字面比对(3-gram)判不了两类情况,NLI 正好补上:
      ① 同义改写:"应当承担赔偿责任" vs "需要负责赔偿" —— 字面零重合,意思一样
      ② 否定反转:"不承担责任"     vs "承担责任"     —— 字面高度重合,意思相反

    ── NLI 实测表现 ──────────────────────────────────────────────
    100 条真实条文 × 4 类样本(见 scratchpad/calib_nli.py)。
    每类样本都有一个"应该被判成什么"的期望标签,百分数 = NLI 判对的比例:

        样本类型         期望标签         NLI 判对
        正例(逐字照抄)   ENTAILMENT       99%
        转述例           ENTAILMENT       99%    ← ① 基本解决
        否定反转         CONTRADICTION    72%    ← ② 大幅改善但不彻底
        负例(条号错配)   NEUTRAL          84%

    ── 为什么不用 bge-reranker(已否决)────────────────────────────
    同一批样本上,交叉编码器给出的相关性分数区间:
        正例      [7.81, 10.31]
        否定反转  [7.77, 10.31]     ← 与正例完全重叠
    两个区间叠在一起,任何阈值都无法把"引对了"和"说反了"分开,
    所以它对否定反转的识别率是 0%。根因:交叉编码器建模的是
    "相关性"(relevance),输出里根本没有"矛盾"这个概念。

    ── 第二道防线 ────────────────────────────────────────────────
    NLI 仍有 28% 的否定反转被判成 ENTAILMENT(说反了却算对),这是法律场景
    最危险的错误。所以再加一道否定词密度差检测。两道合起来把漏网压到 12%,
    代价是 6% 的转述例被误标为待复核 —— 法律场景里漏报比误报危险,这笔划算。
    """
    for c in cites:
        if c.get("verdict") != "uncertain" or not c.get("_claim") or not c.get("_truth_full"):
            continue
        label, conf = nli_judge(c["_truth_full"], c["_claim"], nli_path)
        ndiff = abs(_neg_density(c["_claim"]) - _neg_density(c["_truth_full"]))
        c["nli"] = label
        c["nli_conf"] = round(conf, 3)
        c["neg_diff"] = round(ndiff, 4)

        if conf < min_conf:                      # 模型自己都没把握,不动它
            c["note"] = f"NLI 置信不足({label} {conf:.2f}),维持待复核"
        elif label == "CONTRADICTION":
            c["verdict"] = "misattributed"
            c["note"] = f"NLI 判定矛盾({conf:.2f}) —— 模型把条文说反了"
        elif label == "NEUTRAL":
            c["verdict"] = "misattributed"
            c["note"] = f"NLI 判定无关({conf:.2f}) —— 描述与该条文无关,张冠李戴"
        elif ndiff > neg_th:                     # 第二道防线:NLI 说一致,但否定词分布对不上
            c["note"] = (f"NLI 判一致但否定词分布差异大({ndiff:.3f}) —— "
                         f"疑似把条文说反了,重点人工确认")
        else:
            c["verdict"] = "ok"
            c["note"] = f"字面不重合但 NLI 判定语义蕴含({conf:.2f}),属转述"
    return cites


# ============ 5. 其他客观指标 ============
_HEDGE = ["具体情况具体分析", "咨询专业律师", "以实际情况为准", "视情况而定",
          "建议您咨询", "因人而异", "无法给出", "仅供参考"]
_REFUSE = ["我不能", "我无法", "作为一个", "作为人工智能", "我只是一个"]

def repetition_rate(text, n=8):
    """重复率:重复出现的 n-gram 占比(复读机检测,越低越好)"""
    s = re.sub(r"\s", "", text)
    if len(s) < n * 2:
        return 0.0
    grams = [s[i:i+n] for i in range(len(s) - n + 1)]
    c = Counter(grams)
    return sum(v - 1 for v in c.values()) / len(grams)

def distinct_n(text, n=2):
    """distinct-n:不重复 n-gram 比例(多样性,越高越好)"""
    s = re.sub(r"\s", "", text)
    if len(s) < n:
        return 0.0
    grams = [s[i:i+n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams)

def relevance(question, answer, topk=8):
    """切题率:问题里的实词有多少在回答中出现(粗筛答非所问)"""
    stop = set("的了是吗呢啊怎么如何什么哪些请问我想咨询一下有没有能不能可以需要要求个和与及对在为")
    keys = [c for c in set(re.sub(r"[^一-龥]", "", question)) if c not in stop][:topk*3]
    if not keys:
        return 1.0
    return sum(1 for k in keys if k in answer) / len(keys)

def objective_v2(question, answer, db, nli_path=None):
    """nli_path 传了就启用语义复核:字面比对判不了的 uncertain 交给 NLI 再判一次。
    不传则只做字面比对(快,零模型依赖)。"""
    cites = verify_citations(answer, db)
    if nli_path:
        semantic_recheck(cites, nli_path)     # 只处理 uncertain 那一档,可能把它改判成 ok/misattributed
    bad = [c for c in cites if c["verdict"] in ("out_of_range", "not_found", "misattributed")]
    verified = [c for c in cites if c["verdict"] == "ok"]
    unsure = [c for c in cites if c["verdict"] in ("uncertain", "unverifiable")]
    return {
        "引用条数":   len(cites),
        "核实正确":   len(verified),
        "引用错误":   len(bad),
        "待复核":     len(unsure),
        "语义改判":   sum(1 for c in cites if c.get("nli")),   # 有多少条走了 NLI 复核
        "有引用":     len(cites) > 0,
        "引用全对":   len(cites) > 0 and not bad,
        "法条幻觉":   len(bad) > 0,                                  # 硬伤①:编造/张冠李戴法条
        "复读":       repetition_rate(answer) > 0.15,                # 硬伤②:复读机
        "回避":       any(k in answer for k in _HEDGE),              # 硬伤③:和稀泥
        "拒答":       any(k in answer for k in _REFUSE),
        "过短":       len(answer.strip()) < 30,
        "跑题":       relevance(question, answer) < 0.35,            # 硬伤④:答非所问
        "字数":       len(answer.strip()),
        "distinct2":  round(distinct_n(answer, 2), 3),
        "切题度":     round(relevance(question, answer), 3),
        "_cites":     cites,
    }
