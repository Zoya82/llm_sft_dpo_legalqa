# -*- coding: utf-8 -*-
"""生成 01~05 教学版 notebook（真实数据集 DISC-Law-SFT + 详细易懂注释）。运行：python3 _build_notebooks.py"""
import os, nbformat as nbf

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, text in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(nb, os.path.join(NB_DIR, name))
    print("written:", name)

BOOT = r'''
import os, json, sys   # os=文件/路径操作, json=读写json数据, sys=系统相关
import torch           # PyTorch：深度学习框架(模型和张量都靠它)

# 自动找到项目根目录（notebooks 文件夹的上一级）
ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
DATA = os.path.join(ROOT, "data")      # data 目录完整路径
OUT  = os.path.join(ROOT, "outputs")   # outputs 目录完整路径
os.makedirs(OUT, exist_ok=True)        # 新建 outputs 目录；exist_ok=True=已存在也不报错

# 选计算设备：苹果芯片用 mps，英伟达显卡用 cuda，都没有就用 cpu
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("ROOT  :", ROOT)
print("DEVICE:", DEVICE)
'''.strip()

# 共享：系统提示词 + 取出问题的小函数（02 和 05 都用）
SHARED = r'''
# 系统提示词：告诉模型它是谁、要守什么规矩（每轮对话都带上它）
SYSTEM_PROMPT = "你是专业的中文法律咨询助手。请针对用户的法律问题，给出准确、有依据、条理清晰的解答；涉及具体法条时尽量说明，不要编造不存在的法条或事实。"

# 把一轮完整对话组装成"消息列表"（大模型通用的对话数据格式）
# 每条消息 = 字典 {"role":角色, "content":内容}；role 分 system/user/assistant
def to_messages(system, user, assistant):
    return [
        {"role": "system", "content": system},        # 系统设定
        {"role": "user", "content": user},            # 用户问的
        {"role": "assistant", "content": assistant},  # 模型该答的(=标准答案)
    ]
'''.strip()

# ============================================================== 01
nb01 = [
("md", """# 01 · 任务定义与数据探查（真实数据集）

> 升级版后训练项目的第 1 步。

## 这个项目在做什么
在垂直领域（**中文法律问答**）对 decoder-only 大模型做 **SFT + DPO 对齐**，并建立评测体系。

流程：`01 看数据 → 02 造数据 → 03 SFT → 04 DPO → 05 评测`

## 数据集：DISC-Law-SFT（复旦大学开源，真实数据）
- 来源：[ShengbinYue/DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)，法律问答子集约 **9.3 万条**真实法律咨询问答。
- 字段：`id` / `input`（问题）/ `output`（答案，多为几百字的专业解答，常引用具体法条）。
- 本 notebook 会**自动下载一份样本**（默认 2000 条）到 `data/`，下不动时自动回退到内置 `seed_clauses.jsonl`。

> 任务定义：**输入** = 用户法律问题；**输出** = 准确、有依据的解答。贴你在幂律的法律 AI 经历。

---
💡 注释约定：代码里 `#` 后面是中文说明；不懂的 Python 写法旁边都会点一句它在干嘛。"""),
("code", BOOT),
("code", r'''
# ---- 下载真实数据集样本（首次运行才下载，之后直接用本地文件）----
import urllib.request

RAW = os.path.join(DATA, "disc_law_qa.jsonl")   # 原始数据存这里
N_DOWNLOAD = 2000                                # 只取前 N 条做这个小项目（全量 9.3 万太大）

if not os.path.exists(RAW):                       # 文件不存在才下载
    # 两个地址：国内镜像优先，失败再试官方
    urls = [
        "https://hf-mirror.com/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Pair-QA-released.jsonl",
        "https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT/resolve/main/DISC-Law-SFT-Pair-QA-released.jsonl",
    ]
    for u in urls:
        try:
            print("下载中:", u)
            lines = []
            # urlopen 打开网络连接；for line in r 逐行读（边读边存，读够 N 条就停）
            with urllib.request.urlopen(u, timeout=60) as r:
                for line in r:
                    lines.append(line.decode("utf-8"))   # 网络读到的是字节，decode 转成文字
                    if len(lines) >= N_DOWNLOAD:
                        break
            open(RAW, "w", encoding="utf-8").writelines(lines)   # 写入本地文件
            print("已保存", len(lines), "条到", RAW)
            break                                     # 成功就跳出，不再试下一个地址
        except Exception as e:                        # 任何报错都接住，换下一个地址
            print("失败:", e)
else:
    print("已存在本地数据:", RAW)
'''.strip()),
("code", r'''
# ---- 读取 + 数据清洗 + 去重 → 整理成统一格式 ----
# ★ 二选一：有真实数据就用真实数据，没有(下载失败)才用内置种子集。
#   所以 raw 里要么全是真实数据、要么全是种子数据，绝不会两份掺在一起。
source = RAW if os.path.exists(RAW) else os.path.join(DATA, "seed_clauses.jsonl")
print("数据来源:", os.path.basename(source))

# 逐行读 jsonl（列表推导式：每行 json 文本 → 字典，组成字典列表）
raw = [json.loads(l) for l in open(source, encoding="utf-8") if l.strip()]

pool = []          # 清洗后的干净数据放这
seen = set()       # 记录见过的问题，用来去重（set 判断"在不在里面"很快）
for r in raw:
    # ★ 下面不是"合并两份数据"，而是"兼容两种字段名"：
    #   两个来源字段名不同——真实数据用 input/output，种子集用 question/answer。
    #   A or B 的规则：A 有值就用 A，A 为空/None 才用 B。
    #   所以同一条数据只会命中其中一个键（真实数据命中 input，种子集命中 question），不会混。
    q = (r.get("input") or r.get("question") or "").strip()    # .get(键):有就返回值,没有返回None(比 r["键"] 安全)；末尾 or "" 兜底成空串
    a = (r.get("output") or r.get("answer") or "").strip()
    if not q or not a:                  # 问题或答案为空 → 丢弃
        continue                        # continue=跳过这条，进下一条
    if len(a) < 15 or len(a) > 1000:    # 数据清洗：答案太短(没信息)或太长(训练慢) → 丢弃
        continue
    if q in seen:                       # 这个问题已出现过 → 去重丢弃
        continue
    seen.add(q)                         # 记下这个问题
    pool.append({"question": q, "answer": a})   # 留下，统一成 {question, answer}

QA_POOL = os.path.join(DATA, "qa_pool.json")
json.dump(pool, open(QA_POOL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("清洗后剩余:", len(pool), "条  (原始", len(raw), "条)")
'''.strip()),
("code", r'''
# ---- 看数据画像 + 样例 ----
import statistics as st
ql = [len(x["question"]) for x in pool]    # 每条问题字数
al = [len(x["answer"]) for x in pool]      # 每条答案字数
print("问题长度  平均%.0f  最长%d" % (st.mean(ql), max(ql)))
print("答案长度  平均%.0f  最短%d  最长%d" % (st.mean(al), min(al), max(al)))
print("=" * 60)
for x in pool[:2]:                          # 看前2条真实数据
    print("【问题】", x["question"])
    print("【答案】", x["answer"][:120], "...")   # 答案很长，只看前120字
    print("-" * 60)
'''.strip()),
("md", """## 想一想（面试会问）
1. 为什么要做"数据清洗"（过滤太短/太长、去重）？不做会对训练有什么影响？
2. 真实答案平均几百字、长短不一，这会给训练带来什么挑战？（提示：`max_length` 截断）
3. 这份数据是"问题→答案"，没有给定法条上下文。模型答题时怎么保证有依据、不幻觉？

## 动手改一改
- 把 `N_DOWNLOAD` 调大到 5000 重新下载（先删掉 `data/disc_law_qa.jsonl`），看清洗后能留下多少条。"""),
]

# ============================================================== 02
nb02 = [
("md", """# 02 · 构造 SFT 与 DPO 数据（基于真实数据）

> 对应 JD 关键词：**数据合成 / 偏好数据构造**。

## 两种数据的区别
| | SFT 数据 | DPO 偏好数据 |
|---|---|---|
| 形态 | (问题 → 好答案) | (问题 → 好答案 chosen / 坏答案 rejected) |
| 教模型 | "该怎么答" | "在两个答案里，**为什么这个比那个好**" |

SFT 数据**直接用真实的问答对**；DPO 的 chosen 用真实答案，rejected 用"劣化规则"把真实答案改坏——这是缺少现成偏好数据时的常用做法。"""),
("code", BOOT),
("code", SHARED),
("code", r'''
import random
random.seed(42)                  # 固定随机种子，保证每次划分结果一样(可复现)

# 【配置区】------------------------------------------------------
run_mode = "smoke"               # "smoke"=小数据快速跑通；"full"=用全部数据
pool = json.load(open(os.path.join(DATA, "qa_pool.json"), encoding="utf-8"))   # 读 01 清洗好的数据
random.shuffle(pool)             # 打乱顺序(原地修改 pool)

N_EVAL = 40                                              # 留 40 条做评测
N_SFT  = 200 if run_mode == "smoke" else len(pool)      # SFT 用多少条
N_DPO  = 120 if run_mode == "smoke" else len(pool)      # DPO 用多少条

eval_pool  = pool[:N_EVAL]        # 前 40 条作评测(训练时不用，测真实水平)
train_pool = pool[N_EVAL:]        # 其余作训练池
sft_src = train_pool[:N_SFT]      # 从训练池取前 N_SFT 条做 SFT
dpo_src = train_pool[:N_DPO]      # 取前 N_DPO 条做 DPO
print("可用总数:", len(pool), "| SFT:", len(sft_src), "| DPO:", len(dpo_src), "| eval:", len(eval_pool))
'''.strip()),
("md", "## 2.1 构造 SFT 数据\n真实数据本身就很多样，不需要再增广，直接套成对话格式即可。"),
("code", r'''
# 列表推导式：把每条 {question, answer} 套成 {"messages":[system,user,assistant]}
sft_train = [{"messages": to_messages(SYSTEM_PROMPT, x["question"], x["answer"])} for x in sft_src]
sft_eval  = [{"messages": to_messages(SYSTEM_PROMPT, x["question"], x["answer"])} for x in eval_pool]

# json.dump 写文件：ensure_ascii=False→中文正常显示；indent=2→缩进好看
json.dump(sft_train, open(os.path.join(DATA, "sft_train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(sft_eval,  open(os.path.join(DATA, "sft_eval.json"),  "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("SFT train:", len(sft_train), " eval:", len(sft_eval))
print("样例问题:", sft_train[0]["messages"][1]["content"][:40])
'''.strip()),
("md", """## 2.2 构造 DPO 偏好对
把真实"好答案"用 4 种劣化规则改成"坏答案"，模拟模型常犯的错：
- **截断**(答不完整) · **幻觉**(编造法条) · **回避**(和稀泥) · **事实反转**(把"可以"改"不可以")"""),
("code", r'''
# ---- 4 个劣化函数：各把"好答案 ans"改成一种"坏答案" ----
def corrupt_truncate(ans):
    return ans[: max(10, len(ans) // 2)]   # 截断：只留前一半(// 是整除)，至少留10字

def corrupt_hallucinate(ans):
    return ans[: max(20, len(ans)//2)] + "综上，依据《民法典》第9999条，可直接主张三倍惩罚性赔偿。"  # 幻觉：塞一句编造法条

def corrupt_vague(ans):
    return "这个问题要具体情况具体分析，建议你咨询专业律师，以实际情况为准。"   # 回避：和稀泥

def corrupt_flip(ans):
    out = ans
    # for a,b in [...]：每次取一对词，a=原词 b=反义词，做替换(只替换前几处足以制造矛盾)
    for a, b in [("可以", "不可以"), ("应当", "不应当"), ("有权", "无权"),
                 ("合法", "违法"), ("需要", "不需要"), ("属于", "不属于")]:
        out = out.replace(a, b)            # 字符串.replace(旧, 新)
    return out if out != ans else ("（错误观点）" + ans)   # 没替换成功就加个标记，保证和原答案不同

CORRUPT = [corrupt_flip, corrupt_hallucinate, corrupt_truncate, corrupt_vague]

dpo_rows = []
for x in dpo_src:
    user = x["question"]
    prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    chosen = [{"role": "assistant", "content": x["answer"]}]     # 好答案=真实答案
    for fn in CORRUPT:                      # 遍历4个劣化函数(fn此刻是一个函数)
        bad = fn(x["answer"])              # 调用它，生成坏答案
        if bad.strip() and bad.strip() != x["answer"].strip():   # 确实变坏且非空才要
            dpo_rows.append({"prompt": prompt, "chosen": chosen,
                             "rejected": [{"role": "assistant", "content": bad}]})

json.dump(dpo_rows, open(os.path.join(DATA, "dpo_train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("DPO 偏好对:", len(dpo_rows))
print("chosen  :", dpo_rows[0]["chosen"][0]["content"][:60], "...")
print("rejected:", dpo_rows[0]["rejected"][0]["content"][:60], "...")
'''.strip()),
("md", """## 进阶（可选）：用大模型蒸馏造更高质量的偏好数据
劣化规则造的 rejected 比较"假"。真实项目里常让强模型生成"看起来对但其实有瑕疵"的答案当 rejected（更难、更有训练价值）。需要 API key，默认不跑：
```python
# resp = client.chat.completions.create(model="...", messages=[{"role":"user","content": f"请针对下面法律问题写一个'看似合理但有事实错误'的回答：{q}"}])
# rejected = resp.choices[0].message.content
```

## 想一想
1. 用"劣化真实答案"造 rejected，和"用强模型生成 rejected"，各有什么优缺点？
2. 哪种劣化对法律场景危害最大、最该让模型学会避免？
3. 为什么 eval 的 40 条不能进训练？

## 动手改一改
- 自己加一种劣化规则（如"答非所问"），重跑，到 05 看效果变化。"""),
]

# ============================================================== 03
nb03 = [
("md", """# 03 · SFT 微调（Qwen2.5 + LoRA）

> 对应 JD 关键词：**SFT、PEFT/LoRA**。对应你 T5 项目的"全量微调"——这次换成 **decoder-only 大模型 + LoRA 高效微调**。

## LoRA vs 全量微调
- 全量微调：更新模型**所有**参数，开销大。
- LoRA：冻结原模型，只在每层旁加一对**低秩小矩阵**训练，**只更新约 0.1%~1% 参数**，单卡可跑。
- 面试一句话：为什么用 LoRA？→ 省显存省算力、可叠加多适配器、不破坏原模型能力。"""),
("code", BOOT),
("code", r'''
# 缺包先装（去掉行首 # 运行）：
# %pip install -q peft trl datasets matplotlib

# 【配置区】------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # 底模型；想效果更好改 "Qwen/Qwen2.5-1.5B-Instruct"
run_mode   = "smoke"                          # smoke=快速跑通 / full=正式
EPOCHS     = 1 if run_mode == "smoke" else 3  # 三元表达式：smoke只1轮
LR         = 2e-4                             # 学习率
SFT_OUT    = os.path.join(OUT, "sft_lora")    # 存 LoRA 适配器(小补丁)
SFT_MERGED = os.path.join(OUT, "sft_merged")  # 存"底模+补丁合并后的完整模型"
print("模型:", MODEL_NAME, "| 模式:", run_mode, "| epochs:", EPOCHS)
'''.strip()),
("code", r'''
from transformers import AutoTokenizer, AutoModelForCausalLM   # 自动加载"分词器"和"模型"
from peft import LoraConfig          # LoRA 配置
from trl import SFTTrainer, SFTConfig  # trl：做大模型微调的库
from datasets import load_dataset    # 加载数据集

# 分词器：文字 <-> 数字id 互转（模型只认数字）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # 从网上/本地下载并加载
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 没有填充符就用结束符顶替

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)  # float32 在MPS上最稳
model.to(DEVICE)
print("参数量(亿):", round(sum(p.numel() for p in model.parameters())/1e8, 3))   # numel()=元素个数, /1e8=换算成亿
'''.strip()),
("code", r'''
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,   # r=秩(小矩阵宽度), alpha=缩放, dropout=防过拟合
    target_modules="all-linear",             # 给所有线性层加 LoRA
    task_type="CAUSAL_LM",                   # 任务=因果语言模型(GPT这类生成模型)
)
# 加载 02 生成的训练数据；split="train"取训练划分
train_ds = load_dataset("json", data_files=os.path.join(DATA, "sft_train.json"), split="train")
print("训练样本:", len(train_ds), "| 字段:", train_ds.column_names)
'''.strip()),
("code", r'''
sft_args = SFTConfig(
    output_dir=SFT_OUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,      # 每次喂1条(MPS显存小)
    gradient_accumulation_steps=8,      # 累积8步再更新 → 等效batch=8
    learning_rate=LR,
    logging_steps=5,                    # 每5步打印loss
    save_strategy="no",
    max_length=1024,                    # 真实答案较长，放到1024(超出截断)
    bf16=False, fp16=False,             # MPS用float32
    report_to="none",
)
trainer = SFTTrainer(
    model=model, args=sft_args,
    train_dataset=train_ds,             # 含 messages 字段，trl 自动套对话模板
    peft_config=lora_cfg,
    processing_class=tokenizer,
)
trainer.train()                         # 开始训练(最耗时)
trainer.save_model(SFT_OUT)
print("LoRA 适配器已保存:", SFT_OUT)
'''.strip()),
("code", r'''
# ---- loss 曲线(看有没有学进去) ----
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]   # 让中文正常显示
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
logs = trainer.state.log_history        # 训练日志(列表,每项是字典)
steps = [x["step"] for x in logs if "loss" in x]   # 只挑含loss的记录
loss  = [x["loss"] for x in logs if "loss" in x]
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
plt.figure(figsize=(6,4)); plt.plot(steps, loss, marker="o")
plt.xlabel("step"); plt.ylabel("train loss"); plt.title("SFT loss"); plt.grid(True)
plt.savefig(os.path.join(OUT, "figures", "sft_loss.png"), dpi=120, bbox_inches="tight"); plt.show()
'''.strip()),
("code", r'''
# ---- 合并 LoRA 到底模，导出完整模型(后面DPO/评测用) ----
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)  # 重新加载干净底模
merged = PeftModel.from_pretrained(base, SFT_OUT).merge_and_unload()  # 装上补丁→合并成一体
merged.save_pretrained(SFT_MERGED); tokenizer.save_pretrained(SFT_MERGED)
print("SFT 合并模型已保存:", SFT_MERGED)
'''.strip()),
("code", r'''
# ---- 拿一条 eval 问题，看微调后的回答 ----
def generate(model, tokenizer, system, user, max_new_tokens=256):
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    # apply_chat_template：按该模型规定的对话格式拼文本；add_generation_prompt=True=轮到模型说
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)   # 文本→id张量
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)  # do_sample=False=结果稳定
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()  # 切掉输入,只留生成

sft_eval = json.load(open(os.path.join(DATA, "sft_eval.json"), encoding="utf-8"))   # load读文件(loads读字符串)
sys_p = sft_eval[0]["messages"][0]["content"]
usr_p = sft_eval[0]["messages"][1]["content"]
print("问题:", usr_p)
print("参考:", sft_eval[0]["messages"][2]["content"][:100], "...")
print("SFT后:", generate(merged.to(DEVICE), tokenizer, sys_p, usr_p)[:200])
'''.strip()),
("md", """## 想一想
1. `gradient_accumulation_steps=8` + `batch_size=1` 等效 batch 是多少？为什么单卡要这么配？
2. `max_length=1024` 设小了(比如256)会怎样？（提示：长答案被截断）
3. 为什么训练完要"合并"再用？

## 动手改一改
- `run_mode` 改 `"full"` 或 `r` 改 16，重跑看 loss 和生成质量。"""),
]

# ============================================================== 04
nb04 = [
("md", """# 04 · DPO 偏好对齐

> 对应 JD 关键词：**DPO、偏好优化、对齐**。这是 T5 项目**完全没有**的一环，也是本升级版最大的增量。

## DPO 在做什么
- RLHF 要先训"奖励模型"再用强化学习，链路长、不稳。
- **DPO** 跳过奖励模型，直接用 (chosen, rejected) 偏好对训练：**抬高 chosen 概率、压低 rejected 概率**。
- `beta` 控制"对齐强度 vs 不偏离原模型"的平衡。
- 面试一句话：DPO 把 RLHF 的「奖励建模 + 强化学习」化简成一个可直接优化的分类式损失。"""),
("code", BOOT),
("code", r'''
# 【配置区】------------------------------------------------------
run_mode   = "smoke"
SFT_MERGED = os.path.join(OUT, "sft_merged")     # 03 的产物，DPO 的起点
DPO_OUT    = os.path.join(OUT, "dpo_lora")
DPO_MERGED = os.path.join(OUT, "dpo_merged")     # DPO 合并后的最终模型
EPOCHS     = 1 if run_mode == "smoke" else 3
BETA       = 0.1                                  # 关键超参：越大越听话于偏好、越小越保守
assert os.path.exists(SFT_MERGED), "请先跑完 03 生成 sft_merged"   # assert 条件,提示：不成立就报错
'''.strip()),
("code", r'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(SFT_MERGED, torch_dtype=torch.float32).to(DEVICE)  # 从SFT模型继续

lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
dpo_ds = load_dataset("json", data_files=os.path.join(DATA, "dpo_train.json"), split="train")   # 加载偏好数据
print("DPO 偏好对:", len(dpo_ds), "| 字段:", dpo_ds.column_names)
'''.strip()),
("code", r'''
dpo_args = DPOConfig(
    output_dir=DPO_OUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,                 # DPO 学习率一般比 SFT 更小
    beta=BETA,
    logging_steps=2,
    save_strategy="no",
    max_length=1280, max_prompt_length=256,   # 总长上限 / 问题部分上限(法律问题通常较短)
    bf16=False, fp16=False,
    report_to="none",
)
trainer = DPOTrainer(
    model=model, ref_model=None,        # ref_model=参考模型；用peft时trl自动用合并前基准，无需手动给
    args=dpo_args,
    train_dataset=dpo_ds,               # prompt/chosen/rejected 格式，trl 自动处理
    processing_class=tokenizer,
    peft_config=lora_cfg,
)
trainer.train()
trainer.save_model(DPO_OUT)
print("DPO 适配器已保存:", DPO_OUT)
'''.strip()),
("code", r'''
# ---- loss 曲线 ----
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
logs = trainer.state.log_history
steps = [x["step"] for x in logs if "loss" in x]; loss = [x["loss"] for x in logs if "loss" in x]
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
plt.figure(figsize=(6,4)); plt.plot(steps, loss, marker="o", color="darkorange")
plt.xlabel("step"); plt.ylabel("dpo loss"); plt.title("DPO loss"); plt.grid(True)
plt.savefig(os.path.join(OUT, "figures", "dpo_loss.png"), dpi=120, bbox_inches="tight"); plt.show()
'''.strip()),
("code", r'''
# ---- 合并导出 SFT+DPO 最终模型 ----
base = AutoModelForCausalLM.from_pretrained(SFT_MERGED, torch_dtype=torch.float32)  # 以SFT模型为底
merged = PeftModel.from_pretrained(base, DPO_OUT).merge_and_unload()  # 装DPO补丁并合并
merged.save_pretrained(DPO_MERGED); tokenizer.save_pretrained(DPO_MERGED)
print("最终模型(SFT+DPO)已保存:", DPO_MERGED)
'''.strip()),
("md", """## 想一想
1. `beta` 调大/调小分别会怎样？
2. DPO 为什么需要"参考模型(ref)"？
3. DPO 后若某些问题反而变差，可能是什么原因？（提示：偏好数据质量/覆盖面）

## 动手改一改
- `BETA` 改 0.3 重训，到 05 看 DPO 行指标变化。"""),
]

# ============================================================== 05
nb05 = [
("md", """# 05 · 评测与消融对比

> 对应 JD 关键词：**模型评测、对比实验**。沿用你 T5 项目的"消融对比表 + BLEU"，升级加入 **LLM-as-judge**。

## 三档对比
| 档位 | 模型 |
|---|---|
| base | 原始 Qwen2.5（没微调） |
| +SFT | 03 的 sft_merged |
| +SFT+DPO | 04 的 dpo_merged |

> 开放式问答里答案表述多样，单条 BLEU 偏低很正常；**看三档之间的相对变化**才是重点。"""),
("code", BOOT),
("code", r'''
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {                                # 字典 {名字: 模型路径}
    "base":      "Qwen/Qwen2.5-0.5B-Instruct",
    "+SFT":      os.path.join(OUT, "sft_merged"),
    "+SFT+DPO":  os.path.join(OUT, "dpo_merged"),
}
def generate(model, tokenizer, system, user, max_new_tokens=256):   # 同03,注释见03
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
'''.strip()),
("code", r'''
# 评测集 = 02 留出的 sft_eval（保证和训练不重叠）
sft_eval = json.load(open(os.path.join(DATA, "sft_eval.json"), encoding="utf-8"))
# 列表推导式：从每条的 messages 里取出 system/user/标准答案
eval_items = [{"system": m["messages"][0]["content"],
               "user":   m["messages"][1]["content"],
               "ref":    m["messages"][2]["content"]} for m in sft_eval]
print("评测条数:", len(eval_items))
'''.strip()),
("code", r'''
# ---- 三档模型轮流答题，收集预测 ----
preds = {name: [] for name in MODELS}    # 字典推导式：每个模型名→一个空列表
for name, path in MODELS.items():        # .items() 同时拿 键(name) 和 值(path)
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(DEVICE)
    for it in eval_items:
        preds[name].append(generate(mdl, tok, it["system"], it["user"]))
    del mdl                               # 删模型释放内存(三个轮流跑)
    print(name, "done")
'''.strip()),
("code", r'''
# ---- 字符级 BLEU：衡量生成与标准答案的相似度(0~1)，无需额外装库 ----
import math
from collections import Counter          # Counter=计数器,统计每个元素出现几次

def char_ngrams(s, n):                    # 把字符串切成"连续n个字"的片段并计数
    s = list(s.replace(" ", ""))          # 去空格→单字列表
    return Counter(tuple(s[i:i+n]) for i in range(len(s)-n+1)) if len(s) >= n else Counter()

def bleu_char(ref, hyp, max_n=4):
    scores = []
    for n in range(1, max_n+1):           # 看1/2/3/4字片段重合度
        r, h = char_ngrams(ref, n), char_ngrams(hyp, n)
        overlap = sum((r & h).values())   # r&h=两计数器交集; .values()取次数; sum求和
        scores.append(overlap / max(1, sum(h.values())))   # 命中率(max防除0)
    gm = 0.0 if min(scores) == 0 else math.exp(sum(math.log(x) for x in scores)/len(scores))  # 几何平均
    rl, hl = len(ref), max(1, len(hyp))
    bp = 1.0 if hl >= rl else math.exp(1 - rl/hl)   # 长度惩罚:太短扣分
    return bp * gm

def judge_rule(ref, hyp):                 # 规则版"AI裁判"(粗略),0~1
    if not hyp.strip():
        return 0.0
    score = 0.6                           # 基础分
    if "9999" in hyp or "三倍惩罚性赔偿" in hyp:
        score -= 0.4                      # 出现我们注入的编造法条(幻觉)→扣
    if len(hyp) < 20:
        score -= 0.3                      # 太短/截断→扣
    # 关键词覆盖：标准答案里的高频法律词，生成里命中越多越好(粗略衡量"答到点子上")
    keys = [w for w in ["法律","合同","责任","赔偿","权利","义务","规定","当事人","法院","民法典"] if w in ref]
    hit  = sum(1 for w in keys if w in hyp)
    if keys:
        score += 0.4 * (hit / len(keys))  # 命中比例×0.4
    return max(0.0, min(1.0, score))      # 夹到0~1
'''.strip()),
("code", r'''
import statistics as st
rows = []
for name in MODELS:
    # zip(a,b)=两个列表配对,每次同时取(题目it,预测p)
    bleus  = [bleu_char(it["ref"], p) for it, p in zip(eval_items, preds[name])]
    judges = [judge_rule(it["ref"], p) for it, p in zip(eval_items, preds[name])]
    rows.append((name, st.mean(bleus), st.mean(judges)))

# 拼成 markdown 表格(=简历要引用的消融对比表)
lines = ["| 模型 | 字符BLEU | LLM-judge |", "|---|---|---|"]
for name, b, j in rows:
    lines.append("| %s | %.3f | %.3f |" % (name, b, j))
table = "\n".join(lines)                 # "\n".join(列表)=用换行连成整段
print(table)
open(os.path.join(OUT, "ablation_table.md"), "w", encoding="utf-8").write(table + "\n")
'''.strip()),
("code", r'''
# ---- 消融柱状图 ----
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
names=[r[0] for r in rows]; bl=[r[1] for r in rows]; jg=[r[2] for r in rows]
x=range(len(names)); w=0.35
plt.figure(figsize=(7,4))
plt.bar([i-w/2 for i in x], bl, w, label="字符BLEU")
plt.bar([i+w/2 for i in x], jg, w, label="LLM-judge")
plt.xticks(list(x), names); plt.ylabel("score"); plt.title("base / +SFT / +SFT+DPO 消融对比"); plt.legend()
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
plt.savefig(os.path.join(OUT, "figures", "ablation.png"), dpi=120, bbox_inches="tight"); plt.show()
'''.strip()),
("code", r'''
# ---- badcase：base 答得差、SFT+DPO 修好的例子 ----
os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
bad = []
for i, it in enumerate(eval_items):      # enumerate=同时给下标i和元素it
    b = bleu_char(it["ref"], preds["base"][i])
    d = bleu_char(it["ref"], preds["+SFT+DPO"][i])
    if d - b > 0.05:                     # 最终比base明显好→记为被修好的case
        bad.append({"问题": it["user"], "base": preds["base"][i][:120],
                    "SFT+DPO": preds["+SFT+DPO"][i][:120]})
json.dump(bad, open(os.path.join(OUT, "eval", "badcases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("被修好的 case 数:", len(bad))
for c in bad[:2]:
    print("Q:", c["问题"]); print("base   :", c["base"]); print("SFT+DPO:", c["SFT+DPO"]); print("-"*50)
'''.strip()),
("md", """## 收尾：把它变成你的项目
1. 打开 `README.md`，用**你自己的话**填"五、结论"（数字来自 `ablation_table.md`）。
2. GitHub 只留 notebooks + README + figures + ablation_table（模型权重和数据已被 `.gitignore` 忽略）。
3. 备一段 90 秒口述：做了什么 → SFT 提升多少 → DPO 修好哪类 badcase → 取舍与改进。

## 想一想
1. 开放式问答里 BLEU 偏低，为什么还能用它做"三档对比"？它衡量的到底是什么？
2. 这个规则版 judge 有什么缺陷？换成真大模型当裁判会更好但会带来什么新问题？
3. 上生产还要加哪些评测维度？（延迟、成本、安全合规）"""),
]

build("01_task_and_data_inspection.ipynb", nb01)
build("02_build_sft_dpo_data.ipynb", nb02)
build("03_sft_qwen_lora.ipynb", nb03)
build("04_dpo_alignment.ipynb", nb04)
build("05_eval_and_compare.ipynb", nb05)
print("\n全部生成完成。")
