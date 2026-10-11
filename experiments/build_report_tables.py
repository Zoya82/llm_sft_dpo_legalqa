# -*- coding: utf-8 -*-
"""从原始产物(json / log / trainer_state)直出 PROJECT_REPORT 用的表格,正文复制不手抄。

只读。产物:outputs/report_tables/*.md 与 outputs/report_tables/truth.json(真值字典,供 verify 脚本比对)。
用法:python3 experiments/build_report_tables.py
"""
import os, sys, json, re, statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
EV   = os.path.join(ROOT, "outputs", "eval")
OUTD = os.path.join(ROOT, "outputs", "report_tables")
os.makedirs(OUTD, exist_ok=True)
truth = {}

def pct(x, d=1): return f"{x*100:.{d}f}%"
def save(name, text):
    open(os.path.join(OUTD, name), "w", encoding="utf-8").write(text)
    print("  写出", name)

# ───────── T_sweep:checkpoint sweep 三档八指标 ─────────
sw = json.load(open(f"{EV}/checkpoint_sweep.json", encoding="utf-8"))
keys = ["平均字数","有引用率","引用总数","核实正确","引用准确率","法条幻觉率","复读率","跑题率"]
rows = ["| 指标 | SFT(sft_merged) | DPO ck250 | DPO ck266 |", "|---|---|---|---|"]
for k in keys:
    vals = []
    for tag in ["sft","250","266"]:
        v = sw["结果"][tag][k]
        vals.append(pct(v) if isinstance(v,float) and v<=1 else str(v))
        truth[f"sweep.{tag}.{k}"] = v
    rows.append(f"| {k} | " + " | ".join(vals) + " |")
save("T_sweep.md", "\n".join(rows) + f"\n\n配置:{sw['配置']}\n")

# ───────── T_dpo_curve:266 步完整曲线 ─────────
ts = json.load(open(f"{ROOT}/outputs/dpo_lora/checkpoint-266/trainer_state.json", encoding="utf-8"))
logs = [l for l in ts["log_history"] if "rewards/accuracies" in l]
rows = ["| step | loss | accuracies | margins | chosen 奖励 | rejected 奖励 |", "|---|---|---|---|---|---|"]
for l in logs:
    rows.append(f"| {l['step']} | {l['loss']:.4f} | {l['rewards/accuracies']:.3f} | {l['rewards/margins']:.3f} | {l['rewards/chosen']:.3f} | {l['rewards/rejected']:.3f} |")
    truth[f"dpo266.step{l['step']}"] = {k: l[k] for k in ["loss","rewards/accuracies","rewards/margins","rewards/chosen","rewards/rejected"]}
loss = [l["loss"] for l in logs]; mn = min(loss); i = loss.index(mn)
note = (f"\nmax_steps={ts['max_steps']},global_step={ts['global_step']};loss 最低 {mn:.4f} 在第 {logs[i]['step']} 步,"
        f"末值 {loss[-1]:.4f}(反弹 {(loss[-1]/mn-1)*100:.0f}%);accuracies 起点 {logs[0]['rewards/accuracies']:.3f},"
        f"chosen 奖励 {logs[0]['rewards/chosen']:.2f} → {logs[-1]['rewards/chosen']:.2f}\n")
save("T_dpo_curve.md", "\n".join(rows) + note)

# ───────── T_onpolicy:拒绝采样统计 ─────────
op = json.load(open(f"{EV}/onpolicy_stats.json", encoding="utf-8"))
det = op["detail"]; kept = [d for d in det if d["keep"]]
n_cite = sum(1 for d in kept if d["引用错误"]>0); n_rep = sum(1 for d in kept if d["复读率"]>0.15); n_off = sum(1 for d in kept if d["切题度"]<0.25)
lr = st.mean(d["len_rej"] for d in kept); lc = st.mean(d["len_cho"] for d in kept)
longer = sum(1 for d in kept if d["len_rej"]>d["len_cho"])/len(kept)
per = st.mean(abs(d["len_rej"]-d["len_cho"])/max(1,d["len_cho"]) for d in kept)
t = f"""| 项 | 值 |
|---|---|
| K(每题采样数) | {op['K']} |
| max_new_tokens | {op['max_new']} |
| temperature | {op['temperature']} |
| 源问题数 | {op['n_src']} |
| 产出(保留)条数 | {op['n_kept']}({op['n_kept']/op['n_src']:.0%}) |
| 其中含引用错误 | {n_cite}({n_cite/len(kept):.0%}) |
| 其中复读率 > 0.15 | {n_rep} |
| 其中切题度 < 0.25 | {n_off} |
| rejected 平均字数 / chosen 平均字数 | {lr:.0f} / {lc:.0f} |
| 长度系统性偏差(有符号) | {(lr-lc)/lc:+.1%} |
| 逐条离散(无符号) | {per:.1%} |
| rejected 更长占比 | {longer:.0%} |
"""
truth.update({"onpolicy.n_src":op["n_src"],"onpolicy.n_kept":op["n_kept"],"onpolicy.n_cite":n_cite,"onpolicy.len_rej":lr,"onpolicy.len_cho":lc,"onpolicy.longer":longer})
save("T_onpolicy.md", t)

# ───────── T_forget:遗忘探针六档 ─────────
fp = json.load(open(f"{EV}/forgetting_probe.json", encoding="utf-8"))
rows = ["| scale | scaling | ≈α | A 组分点率 | B 组分点率 | A 组字数 | 法律术语密度 | 引用准确率 | 引用数 | 幻觉率 |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in fp["扫描"]:
    rows.append(f"| {r['scale']} | {r['scaling']:.2f} | {r['alpha']:.0f} | {pct(r['A组分点率'],0)} | {pct(r['B组分点率'],0)} | {r['A组字数']:.0f} | {r['法律术语密度']:.2f} | {pct(r['引用准确率'],0)} | {r['引用数']} | {pct(r['幻觉率'],0)} |")
    truth[f"forget.scale{r['scale']}"] = r
save("T_forget.md", "\n".join(rows) + f"\n\nLoRA 配置:{fp['LoRA配置']};探针 A {len(fp['探针A'])} 条 / B {len(fp['探针B'])} 条\n")

# ───────── T_precision:bf16 vs fp32 ─────────
pa = json.load(open(f"{EV}/precision_ab.json", encoding="utf-8"))
a, b = pa["结果"]["fp32"], pa["结果"]["bf16"]
t = f"""| | fp32 | bf16 | 变化 |
|---|---|---|---|
| 秒/步 | {a['秒/步']:.1f} | {b['秒/步']:.1f} | 快 {(a['秒/步']/b['秒/步']-1)*100:.0f}% |
| 总耗时(分,{pa['steps']} 步) | {a['总耗时分']:.1f} | {b['总耗时分']:.1f} | — |
| 峰值 alloc 显存(GB) | {a['峰值_alloc_GB']:.2f} | {b['峰值_alloc_GB']:.2f} | 省 {(1-b['峰值_alloc_GB']/a['峰值_alloc_GB'])*100:.0f}% |
| 峰值 driver 显存(GB) | {a['峰值_driver_GB']:.2f} | {b['峰值_driver_GB']:.2f} | — |
| 出现 NaN | {a['有NaN']} | {b['有NaN']} | — |
| LoRA 参数 dtype | {a['LoRA参数dtype']} | {b['LoRA参数dtype']} | PEFT 保持 fp32 |

| step | fp32 loss | bf16 loss | 差值(bf16−fp32) | fp32 acc | bf16 acc |
|---|---|---|---|---|---|
"""
d = []
for x, y in zip(a["logs"], b["logs"]):
    d.append(y["loss"]-x["loss"])
    t += f"| {x['step']} | {x['loss']:.4f} | {y['loss']:.4f} | {y['loss']-x['loss']:+.4f} | {x['acc']:.3f} | {y['acc']:.3f} |\n"
same = sum(1 for v in d if v > 0)
t += f"\nloss 平均绝对偏差 {st.mean(abs(v) for v in d):.4f};{same}/{len(d)} 个点 bf16 更高(方向一致)\n"
truth.update({"prec.fp32.spd":a['秒/步'],"prec.bf16.spd":b['秒/步'],"prec.fp32.acc50":a['logs'][-1]['acc'],"prec.bf16.acc50":b['logs'][-1]['acc']})
save("T_precision.md", t)

# ───────── T_sftq:SFT 数据质量打分 ─────────
sq = json.load(open(f"{EV}/sft_quality.json", encoding="utf-8"))
rows = sq["rows"]; sc = sorted(r["score"] for r in rows)
q = lambda p: sc[int(len(sc)*p)]
cur = [r for r in rows if r["in_current_2000"]]; top = sorted(rows, key=lambda r:-r["score"])[:2000]
full = sq["满分构成"]
def flag(rs, f): return sum(1 for r in rs if f(r))/len(rs)
t = f"""| 项 | 值 |
|---|---|
| 打分器版本 | {sq['版本']} |
| 满分构成 | {full} |
| 样本数 | {sq['n']} |
| 中位 / 均值 | {sq['中位']} / {sq['均值']} |
| 分位 p5 / p25 / p50 / p75 / p95 | {q(.05)} / {q(.25)} / {q(.5)} / {q(.75)} / {q(.95)} |
| 不同分值个数 | {len(set(sc))} |
| 最常见 6 个分值覆盖 | {sum(v for _,v in Counter(sc).most_common(6))/len(sc):.0%} |
| 有引用的条数 | {sum(1 for r in rows if r['引用对']+r['引用错']>0)}({flag(rows, lambda r: r['引用对']+r['引用错']>0):.1%}) |
| 引用对 / 引用错 → 训练数据引用准确率 | {sum(r['引用对'] for r in rows)} / {sum(r['引用错'] for r in rows)} → {sum(r['引用对'] for r in rows)/(sum(r['引用对'] for r in rows)+sum(r['引用错'] for r in rows)):.1%} |
| 冗余尾巴(裸标题类)占比 | {flag(rows, lambda r: r['冗余尾巴']):.2%} |
| 字数中位 | {st.median(r['字数'] for r in rows):.0f} |

回答形态分布:{dict(Counter(r['回答形态'] for r in rows).most_common())}
问题类型分布:{dict(Counter(r['问题类型'] for r in rows).most_common())}

| 同样取 2000 条 | 当前(shuffle 前 2000) | 按质量 top-2000 |
|---|---|---|
| 平均分 | {st.mean(r['score'] for r in cur):.1f} | {st.mean(r['score'] for r in top):.1f} |
"""
for k in full:
    t += f"| {k}(均分/满分 {full[k]}) | {st.mean(r['detail'][k] for r in cur):.1f} | {st.mean(r['detail'][k] for r in top):.1f} |\n"
for nm, f in [("和稀泥(坏)", lambda r:r['回答形态']=='和稀泥'),("没答到点上(坏)", lambda r:r['回答形态']=='没答到点上'),("引用有错(坏)", lambda r:r['引用错']>0),("超 500 字(坏)", lambda r:r['字数']>500)]:
    t += f"| {nm} | {flag(cur,f):.0%} | {flag(top,f):.0%} |\n"
t += f"| 两批重合 | {len(set(r['idx'] for r in cur)&set(r['idx'] for r in top))/2000:.0%} | top-2000 最低分 {min(r['score'] for r in top)} |\n"
save("T_sftq.md", t)

# ───────── T_labels:人工 38 条 ─────────
lab = json.load(open(f"{EV}/labels/batch1_part1.json", encoding="utf-8"))
lab = {k:v for k,v in lab.items() if not k.startswith("_")}
byidx = {str(r["idx"]): r for r in rows}
pairs = [(byidx[k]["score"], v["score"]) for k,v in lab.items()]
xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]; mx,my=st.mean(xs),st.mean(ys)
r_ = sum((a-mx)*(b-my) for a,b in zip(xs,ys))/((sum((a-mx)**2 for a in xs)**.5)*(sum((b-my)**2 for b in ys)**.5))
import math
z = 0.5*math.log((1+r_)/(1-r_)); se=1/math.sqrt(len(pairs)-3)
lo,hi = math.tanh(z-1.96*se), math.tanh(z+1.96*se)
t = f"| 人工分 | 条数 | 占比 | 规则分中位 |\n|---|---|---|---|\n"
for s_ in [5,4,3,2,1]:
    g=[p[0] for p in pairs if p[1]==s_]
    t += f"| {s_} | {len(g)} | {len(g)/len(pairs):.0%} | {st.median(g) if g else '—'} |\n"
t += f"\nn={len(pairs)};Pearson r = {r_:.3f},95% CI [{lo:+.2f}, {hi:+.2f}];direct=True {sum(1 for v in lab.values() if v['direct'])}/{len(lab)}\n"
truth.update({"labels.n":len(pairs),"labels.r":r_})
save("T_labels.md", t)

# ───────── T_ablation148:148 条三档/四档 重算 ─────────
import legal_check as L
db = L.load_law_db(f"{ROOT}/data/stard_corpus.jsonl", verbose=False)
pc = json.load(open(f"{EV}/preds_cache.json", encoding="utf-8"))
items = pc["items"]; preds = dict(pc["preds"])
preds["rag"] = json.load(open(f"{EV}/preds_rag.json", encoding="utf-8"))
preds["rag_v1"] = json.load(open(f"{EV}/preds_rag_v1.json", encoding="utf-8"))
preds["rag_v3"] = json.load(open(f"{EV}/preds_rag_v3.json", encoding="utf-8"))
preds["rag_sft"] = json.load(open(f"{EV}/preds_rag_sftbase.json", encoding="utf-8"))
preds["sft_rp"] = json.load(open(f"{EV}/preds_sft_rp.json", encoding="utf-8"))
def ev(ps):
    obs=[L.objective_v2(it["user"], p, db) for it,p in zip(items, ps)]
    ok=sum(o["核实正确"] for o in obs); bad=sum(o["引用错误"] for o in obs); unc=sum(o["待复核"] for o in obs)
    return dict(字数=round(st.mean(len(p) for p in ps)), 有引用率=st.mean(o["有引用"] for o in obs), 引用总数=ok+bad+unc,
                核实正确=ok, 确认错误=bad, 待复核=unc, 引用准确率=ok/max(1,ok+bad), 幻觉率=st.mean(o["法条幻觉"] for o in obs),
                复读率=st.mean(o["复读"] for o in obs), 跑题率=st.mean(o["跑题"] for o in obs), 切题度=st.mean(o["切题度"] for o in obs),
                含幻觉条数=sum(1 for o in obs if o["法条幻觉"]), 有引用条数=sum(1 for o in obs if o["有引用"]))
names = {"base":"base","sft":"+SFT","dpo":"+SFT+DPO(规则 450 步)","rag":"+SFT+DPO+RAG(v2)","rag_v1":"RAG v1(封闭式 k=3)","rag_v3":"RAG v3(强制标注)","rag_sft":"RAG(sft_merged 基座)","sft_rp":"+SFT(rp=1.15)"}
rows_ = ["| 档位 | 字数 | 有引用率 | 引用总数 | 核实正确 | 确认错误 | 待复核 | 引用准确率 | 法条幻觉率 | 复读率 | 跑题率 | 切题度 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k in ["base","sft","dpo","rag","rag_v1","rag_v3","rag_sft","sft_rp"]:
    e = ev(preds[k]); truth[f"ab148.{k}"] = e
    rows_.append(f"| {names[k]} | {e['字数']} | {pct(e['有引用率'],0)} | {e['引用总数']} | {e['核实正确']} | {e['确认错误']} | {e['待复核']} | {pct(e['引用准确率'],0)} | {pct(e['幻觉率'],0)} | {pct(e['复读率'],0)} | {pct(e['跑题率'],0)} | {e['切题度']:.3f} |")
    print("   ", names[k], "done")
save("T_ablation148.md", "\n".join(rows_) + "\n\n注:用当前 legal_check(含 NLI 复核)重算;BLEU/ROUGE 见 outputs/ablation_table.md。\n")

# ───────── T_blind:盲评映射核对 ─────────
bm = json.load(open(f"{EV}/blind/rag_ab_mapping.json", encoding="utf-8"))
votes = "1A 2A 3B 4B 5B 6B 7B 8A 9A 10B 11B 12B 13A 14B 15B 16B 17A 18A 19A 20A 21B 22A 23A 24A 25B 26A 27B 28B 29A 30B 31B 32B 33A 34A 35A 36A 37B 38A 39B 40A 41B 42B 43A 44B 45B 46B 47B 48A 49A 50A 51A 52B 53A 54A 55A 56A 57B 58A 59A 60A".split()
win=Counter(bm[v[:-1]][v[-1]] for v in votes); posA=sum(1 for v in votes if v[-1]=="A")
p=win["sft"]/60; se=math.sqrt(p*(1-p)/60)
t=f"| | 胜出 | 胜率 | 95% CI |\n|---|---|---|---|\n| SFT+RAG | {win['sft']} | {p:.0%} | [{p-1.96*se:.0%}, {p+1.96*se:.0%}] |\n| SFT+DPO+RAG | {win['dpo']} | {1-p:.0%} | — |\n\n判 A 位胜 {posA/60:.0%};SFT 版排在 A 位 {sum(1 for m in bm.values() if m['A']=='sft')/60:.0%}\n"
truth.update({"blind4.sft":win["sft"],"blind4.dpo":win["dpo"]})
save("T_blind_rag.md", t)

json.dump(truth, open(os.path.join(OUTD,"truth.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("完成 →", OUTD)
