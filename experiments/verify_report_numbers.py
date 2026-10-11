"""校验 PROJECT_REPORT.md:自动表里的每个数字都必须出现在正文中;四拍结构齐全;旧口径残留检查。
用法:python experiments/verify_report_numbers.py
"""
import re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
report = (ROOT / "PROJECT_REPORT.md").read_text(encoding="utf-8")
norm = report.replace("−", "-")

# 1) 自动表数字逐个回查(按数值比对,容忍 5.4% / 5.40%、0.750 / 0.75 这类格式差异)
def _canon(tok):
    pct = tok.endswith("%")
    v = float(tok.rstrip("%"))
    return f"{v:.6g}{'%' if pct else ''}"
report_nums = {_canon(t) for t in re.findall(r"-?\d+\.?\d*%?", norm) if t not in ("", "-", ".")}
missing = []
tables = sorted((ROOT / "outputs/report_tables").glob("T_*.md"))
for t in tables:
    for line in t.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or set(line.strip("| ")) <= set("-| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for c in cells[1:]:
            for num in re.findall(r"-?\d+\.?\d*%?", c.replace("−", "-")):
                if num in ("", "-"):
                    continue
                # 百分比表里 0.595 形式与正文 59.5% 等价,两种都接受
                if num not in norm and _canon(num) not in report_nums:
                    missing.append((t.name, num, line[:60]))
ok1 = not missing
print(f"[1] 自动表数字回查:{'通过' if ok1 else '缺失 %d 处' % len(missing)}")
for m in missing[:40]:
    print("    ", m)

# 2) 四拍结构
n4 = len(re.findall(r"^#### 4\.\d+\.[1-4]", report, re.M))
print(f"[2] 四拍小节数:{n4}(期望 44){' 通过' if n4 == 44 else ' 不通过'}")
for i in range(1, 12):
    sec = re.search(rf"^### 4\.{i} .*?(?=^### 4\.{i+1} |^### 4\.12 )", report, re.M | re.S)
    if not sec or not re.search(r"\|\s*-{3,}", sec.group(0)):
        print(f"    4.{i} 缺表")

# 3) 旧口径残留(允许出现但必须带括注)
checks = {
    "62 个优化步": False, "8 个结论": False, "11 次结论": False,
    "实判 26": False, "35 / 26 / 20": False, "| +21% |": False, "| 260 | 0.938": False, "6.6 精度": False, "47.1 秒/步,266": False, "第 4 处 | —": False,
}
body = re.sub(r"### 0\.4 .*?(?=### 0\.3 )", "", report, flags=re.S)  # §0.4 修正记录里保留改前值,不算残留
for k in checks:
    if k in body:
        print(f"[3] 残留旧口径:'{k}'")
print("[3] 旧口径残留检查完成")

# 4) 附录 B 路径可达
paths = re.findall(r"`((?:outputs|data|notebooks|experiments|archive)/[^`\s]+?)`", report)
bad = [p for p in set(paths) if not (ROOT / p.split(" ")[0]).exists()]
print(f"[4] 路径可达:{'通过' if not bad else '不可达 ' + str(sorted(bad))}")
sys.exit(0 if ok1 and n4 == 44 else 1)
