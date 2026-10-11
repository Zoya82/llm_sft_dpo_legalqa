# -*- coding: utf-8 -*-
"""灾难性遗忘评测:领域外探针集 + LoRA 缩放扫描

【v2 修正】v1 把所有探针的"分点率"混在一起算平均,这个数说明不了问题 ——
有些问题本来就该分点回答("列表和元组有什么区别""推荐几部电影"),分点是合理的;
只有在**明确限制了格式**的题目上("用三句话""一句话总结"),分点才是行为漂移的证据。
v2 把探针分成两组分别统计:

  A 组 受限格式(24 条):问题明确要求"三句话/一句话/一个词",分点 = 违背指令
  B 组 开放格式(24 条):问题适合列举,分点 = 合理表达

漂移的正确度量是 **A 组分点率**(越低越好),B 组分点率只作参照。

【缩放的含义】LoRA 输出是 (α/r)·BA·x,本项目 r=8、α=16 → scaling=2.0。
脚本里的 scale 是**在此基础上的缩放系数**,不是 α 本身:
    scale 1.00 → scaling 2.0(等价 α=16,即训练时的配置)
    scale 0.25 → scaling 0.5(等价 α=4)
    scale 0.00 → scaling 0  (等价关掉 adapter = 底模,作为锚点)
权重空间上等价于线性插值 W' = W_base + scale·ΔW。
"""
from pathlib import Path
import json, re, sys, torch, statistics as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import legal_check as L
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = str(Path(__file__).resolve().parents[1])
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------- A 组:明确限制格式 → 分点 = 违背指令 ----------
PROBES_A = [
    "帮我写一封三句话的请假邮件,理由是家里有事",
    "用一句话解释什么是缓存",
    "把'尽快回复我'改写成更礼貌的说法,只给改写结果",
    "用一个词形容今天下雨的心情",
    "写一个五个字的标题,主题是春天",
    "用一句话总结:项目延期了,但预算没超",
    "给同事发一条微信,说会议改到明天下午三点,一句话即可",
    "用两句话介绍一下杭州",
    "把这句话改成被动语态,只回答改写结果:小明打碎了杯子",
    "用一句话说明为什么要备份数据",
    "给这段话起个标题,不超过八个字:公司决定下周开始远程办公",
    "用一句话安慰一个考试没考好的朋友",
    "把'我不同意'换成委婉的说法,只给一种",
    "用一句话解释什么是复利",
    "帮我写一句生日祝福,不要超过十五个字",
    "用一句话说明早睡的好处",
    "把这个需求缩成一句话:希望网站加载更快、界面更简洁、支持手机访问",
    "用一个成语形容做事很有条理",
    "写一句话的会议纪要:确定了下季度目标,由张三负责跟进",
    "用一句话回答:番茄是水果还是蔬菜?",
    "帮我拟一个不超过十个字的活动名称,主题是读书分享",
    "用一句话解释什么是通货膨胀",
    "把'这个方案不行'改得更专业一些,只给一句",
    "用一句话描述你眼中的秋天",
]

# ---------- B 组:开放式,适合列举 → 分点合理 ----------
PROBES_B = [
    "Python 里列表和元组有什么区别?",
    "推荐几部适合周末看的电影",
    "新手健身该怎么入门?",
    "为什么天空是蓝色的?",
    "简单说说光合作用的原理",
    "for 循环和 while 循环该怎么选?",
    "什么是 API?举个例子",
    "数据库索引为什么能加快查询?",
    "解释一下什么叫递归",
    "咖啡因为什么能提神?",
    "为什么飞机能飞起来?",
    "地震是怎么形成的?",
    "我今天面试没通过,有点难受",
    "最近总是失眠,有什么办法吗?",
    "朋友生日快到了,送什么礼物好?",
    "工作压力好大,怎么放松一下?",
    "怎么提高英语口语?",
    "第一次去日本旅行要注意什么?",
    "如何培养每天阅读的习惯?",
    "新手学做菜从哪几道菜开始比较好?",
    "怎么挑选一台适合办公的笔记本电脑?",
    "养猫需要准备哪些东西?",
    "如何整理房间才能保持整洁?",
    "怎么开始写技术博客?",
]

SYSTEM = "你是一个乐于助人的助手。"      # 通用 system,不带法律助手人设
LEGAL_SYS = ("你是专业的中文法律咨询助手。请针对用户的法律问题，给出准确、有依据、条理清晰的解答；"
             "涉及具体法条时尽量说明，不要编造不存在的法条或事实。")

_LIST_HEAD = re.compile(r"^\s*(?:[1-9１-９一二三四五六七八九][、.\.)）]|\([1-9]\)|（[1-9]）|[-*•])")
_LEGAL = re.compile(r"法律|法规|条例|司法解释|《[^》]{2,20}法[^》]{0,6}》|"
                    r"第[一二三四五六七八九十百千零〇\d]{1,10}条|人民法院|诉讼|违法|民法典|刑法")

def is_listy(t):
    return bool(_LIST_HEAD.match(t.strip()))

def legal_density(t):
    t = t.strip()
    return len(_LEGAL.findall(t)) / max(1, len(t)) * 100

@torch.no_grad()
def gen(model, tok, system, user, max_new_tokens=200):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    i = tok(t, return_tensors="pt").to(model.device)
    o = model.generate(**i, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(o[0][i["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def set_scale(peft_model, scale):
    """把所有 LoRA 层的 scaling 乘以 scale(首次调用时记住原值)"""
    for mod in peft_model.modules():
        if hasattr(mod, "scaling") and isinstance(getattr(mod, "scaling", None), dict):
            if not hasattr(mod, "_base_scaling"):
                mod._base_scaling = dict(mod.scaling)
            for k in mod.scaling:
                mod.scaling[k] = mod._base_scaling[k] * scale


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen2.5-0.5B-Instruct")
    db = L.load_law_db(f"{ROOT}/data/stard_corpus.jsonl", verbose=False)
    eval_items = [{"user": m["messages"][1]["content"]}
                  for m in json.load(open(f"{ROOT}/data/sft_eval.json", encoding="utf-8"))][:40]

    base = AutoModelForCausalLM.from_pretrained(f"{ROOT}/models/Qwen2.5-0.5B-Instruct",
                                                dtype=torch.float32).to(DEV)
    peft = PeftModel.from_pretrained(base, f"{ROOT}/outputs/sft_lora").eval()

    SCALES = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0]     # 含 0(=底模)作为锚点
    print("LoRA 配置:r=8, α=16 → scaling=2.0(scale=1.0 时)")
    print("A 组=明确限制格式(分点即违背指令) | B 组=开放式(分点合理)\n")
    print(f"{'scale':>6}{'scaling':>9}{'≈α':>5} | {'A组分点率':>10}{'B组分点率':>10}{'A组字数':>9}"
          f"{'法律味':>7} | {'引用准确率':>11}{'引用数':>7}{'幻觉率':>7}")
    print("-" * 92)
    rows = []
    for s in SCALES:
        set_scale(peft, s)
        a_ans = [gen(peft, tok, SYSTEM, q) for q in PROBES_A]
        b_ans = [gen(peft, tok, SYSTEM, q) for q in PROBES_B]
        legal = [gen(peft, tok, LEGAL_SYS, it["user"], 512) for it in eval_items]
        ob = [L.objective_v2(it["user"], p, db) for it, p in zip(eval_items, legal)]
        ok = sum(o["核实正确"] for o in ob); bad = sum(o["引用错误"] for o in ob)
        r = {"scale": s, "scaling": 2.0*s, "alpha": 16*s,
             "A组分点率": st.mean([is_listy(x) for x in a_ans]),
             "B组分点率": st.mean([is_listy(x) for x in b_ans]),
             "A组字数": st.mean([len(x) for x in a_ans]),
             "法律术语密度": st.mean([legal_density(x) for x in a_ans + b_ans]),
             "引用准确率": ok/(ok+bad) if ok+bad else 0,
             "引用数": sum(o["引用条数"] for o in ob),
             "幻觉率": st.mean([o["法条幻觉"] for o in ob]),
             "_A样例": a_ans[:3]}
        rows.append(r)
        print(f"{s:>6.3f}{2.0*s:>9.2f}{16*s:>5.0f} | {r['A组分点率']:>10.0%}{r['B组分点率']:>10.0%}"
              f"{r['A组字数']:>9.0f}{r['法律术语密度']:>7.2f} | "
              f"{r['引用准确率']:>11.0%}{r['引用数']:>7}{r['幻觉率']:>7.0%}", flush=True)

    json.dump({"说明": "A组=明确限制格式(分点即违背指令);B组=开放式(分点合理)",
               "LoRA配置": {"r": 8, "lora_alpha": 16, "base_scaling": 2.0},
               "探针A": PROBES_A, "探针B": PROBES_B, "扫描": rows},
              open(f"{ROOT}/outputs/eval/forgetting_probe.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n结果已存 outputs/eval/forgetting_probe.json")
    print("\nA 组样例(限制格式的题目,看它守不守指令):")
    for r in rows:
        if r["scale"] in (1.0, 0.25, 0.0):
            print(f"\n--- scale={r['scale']} (≈α={r['alpha']:.0f}) ---")
            for q, a in zip(PROBES_A[:3], r["_A样例"]):
                print(f"  Q: {q}")
                print(f"  A: {a[:110]}")
