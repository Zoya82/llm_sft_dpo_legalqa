# -*- coding: utf-8 -*-
"""MPS 上 fp32 vs bf16 的真实训练对比 —— 验证"MPS 必须用 fp32"这条结论还成不成立。

【为什么要跑这个】
项目里 03/04 写死了 `dtype=torch.float32  # float32 在 MPS 上最稳`。
后来讨论 1.5B 时提到改 bf16 能把内存腰斩,但当时我只测了**一个算子**:

    x = torch.randn(64,64,dtype=torch.bfloat16,device='mps'); (x@x).sum()   # 能跑

**矩阵乘能跑 ≠ 训练稳定。** 一次训练涉及几十种算子(layernorm / softmax /
attention / 交叉熵 / 全套反向传播),任何一个在 bf16 下有问题都会出事。
所以这里跑真实训练来验,而不是测算子。

【为什么 DPO 是最严的测试场景】
DPO 的 loss 是四个 log-prob 的组合运算,而本项目实测 `rewards/rejected` 到过 **−8.27**、
`margins` 到 **6.53** —— 这是**大数相减**。bf16 尾数只有 7 位(约 3 位十进制有效数字),
fp32 有 23 位。如果 bf16 在这里都稳,那 SFT 场景更没问题。

【实验设计】
唯一变量是 dtype,其余全部对齐:同 seed / 同 data_seed / 同数据 / 同步数。
顺序跑(不并行),每组跑完释放显存再跑下一组。

三个问题一次回答:
    ① 数值稳定吗   —— 两条 loss 曲线是否贴合,有没有 NaN
    ② 快多少       —— 秒/步(此前"速度接近翻倍"的说法待验)
    ③ 省多少内存   —— torch.mps.current_allocated_memory()

【用法】
    python3 experiments/precision_ab.py            # 默认 50 步
    python3 experiments/precision_ab.py 30         # 想快点就少跑几步
产物:outputs/eval/precision_ab.json
"""
import os, sys, json, time, gc
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
DEV  = "mps" if torch.backends.mps.is_available() else "cpu"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEED  = 42


def mem():
    """MPS 侧的真实占用(ps 的 RSS 看不到这部分)"""
    if DEV != "mps":
        return 0.0, 0.0
    return (torch.mps.current_allocated_memory() / 2**30,
            torch.mps.driver_allocated_memory() / 2**30)


def run(dtype, tag):
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset

    sft = os.path.join(OUT, "sft_merged")
    tok = AutoTokenizer.from_pretrained(sft)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(sft, dtype=dtype).to(DEV)
    ds = load_dataset("json", data_files=os.path.join(DATA, "dpo_train.json"), split="train")

    peak = {"alloc": 0.0, "driver": 0.0}

    class MemProbe(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            a, d = mem()
            peak["alloc"] = max(peak["alloc"], a)
            peak["driver"] = max(peak["driver"], d)

    args = DPOConfig(
        output_dir=os.path.join(OUT, f"_prec_{tag}"),
        max_steps=STEPS, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=5e-5, beta=0.1, logging_steps=5,
        save_strategy="no", max_length=768, gradient_checkpointing=True,
        dataloader_pin_memory=False,
        bf16=False, fp16=False,      # 【关键】不走 autocast+GradScaler 路径(MPS 上不稳),
                                     # 精度完全由上面 from_pretrained 的 dtype 决定
        report_to="none", seed=SEED, data_seed=SEED,
    )
    tr = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                    processing_class=tok,
                    peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules="all-linear", task_type="CAUSAL_LM"))
    tr.add_callback(MemProbe())

    # 记录 LoRA 参数的实际 dtype —— PEFT 会跟随基座,这一点要核实而不是假设
    lora_dt = {str(p.dtype) for n, p in tr.model.named_parameters() if "lora_" in n}

    t0 = time.time()
    torch.manual_seed(SEED)
    tr.train()
    el = time.time() - t0

    logs = [{"step": l["step"], "loss": l["loss"],
             "acc": l["rewards/accuracies"], "margin": l["rewards/margins"],
             "chosen": l["rewards/chosen"], "rejected": l["rewards/rejected"]}
            for l in tr.state.log_history if "rewards/accuracies" in l]

    r = {"dtype": str(dtype), "秒/步": el / STEPS, "总耗时分": el / 60,
         "LoRA参数dtype": sorted(lora_dt),
         "峰值_alloc_GB": peak["alloc"], "峰值_driver_GB": peak["driver"],
         "有NaN": any(l["loss"] != l["loss"] for l in logs), "logs": logs}

    del tr, model
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    time.sleep(3)
    return r


if __name__ == "__main__":
    print(f"MPS 精度对比:fp32 vs bf16,各跑 {STEPS} 步(同 seed / 同数据 / 唯一变量是 dtype)")
    print(f"预计每组 {STEPS*52/60:.0f} 分钟\n", flush=True)

    res = {}
    for dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        print(f"───── {tag} 开始 ─────", flush=True)
        res[tag] = run(dtype, tag)
        r = res[tag]
        print(f"  完成:{r['秒/步']:.1f} 秒/步 | 峰值 alloc {r['峰值_alloc_GB']:.2f} GB "
              f"| LoRA dtype {r['LoRA参数dtype']} | NaN={r['有NaN']}\n", flush=True)

    a, b = res["fp32"], res["bf16"]
    print("=" * 70)
    print(f"{'':<16}{'fp32':>12}{'bf16':>12}{'变化':>14}")
    print("-" * 70)
    print(f"{'秒/步':<16}{a['秒/步']:>12.1f}{b['秒/步']:>12.1f}"
          f"{(a['秒/步']/b['秒/步']-1)*100:>13.0f}%")
    print(f"{'峰值显存 GB':<16}{a['峰值_alloc_GB']:>12.2f}{b['峰值_alloc_GB']:>12.2f}"
          f"{(1-b['峰值_alloc_GB']/max(1e-9,a['峰值_alloc_GB']))*100:>13.0f}%")
    print(f"{'出现 NaN':<16}{str(a['有NaN']):>12}{str(b['有NaN']):>12}")
    print()
    print(f"{'step':>6}{'fp32 loss':>12}{'bf16 loss':>12}{'差值':>10}"
          f"{'fp32 acc':>11}{'bf16 acc':>11}")
    print("-" * 70)
    for x, y in zip(a["logs"], b["logs"]):
        print(f"{x['step']:>6}{x['loss']:>12.4f}{y['loss']:>12.4f}"
              f"{y['loss']-x['loss']:>+10.4f}{x['acc']:>11.3f}{y['acc']:>11.3f}")

    d = [abs(y["loss"] - x["loss"]) for x, y in zip(a["logs"], b["logs"])]
    md = sum(d) / len(d)
    print(f"\nloss 平均绝对偏差 {md:.4f}")
    print("判读:")
    if b["有NaN"]:
        print("  ❌ bf16 出现 NaN → 不可用,项目里的 fp32 保持不动")
    elif md < 0.02:
        print("  ✅ 两条曲线贴合(偏差 <0.02)→ bf16 在本任务上可用")
    elif md < 0.05:
        print("  ⚠️ 有可见偏差(0.02~0.05)→ 短程可用,长程需再验")
    else:
        print("  ❌ 偏差过大(>0.05)→ bf16 数值不可靠,继续用 fp32")

    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    json.dump({"steps": STEPS, "seed": SEED, "结果": res},
              open(os.path.join(OUT, "eval", "precision_ab.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    import shutil
    for tag in ("fp32", "bf16"):
        shutil.rmtree(os.path.join(OUT, f"_prec_{tag}"), ignore_errors=True)
    print("\n已存 outputs/eval/precision_ab.json")
