import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============ 读取数据 ============
df = pd.read_csv("results/raw_results.csv")

# 保证难度按 simple -> medium -> hard 排序
DIFF_ORDER = ["simple", "medium", "hard"]
DIFF_LABELS = {"simple": "简单题", "medium": "中等题", "hard": "困难题"}
df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFF_ORDER, ordered=True)

# ============ 1. 按难度 + 条件的汇总 ============
summary = df.groupby(["difficulty", "condition"], observed=True).agg(
    accuracy=("correct", "mean"),
    avg_latency=("latency", "mean"),
    avg_completion_tokens=("completion_tokens", "mean"),
    count=("correct", "count"),
).reset_index()

summary.to_csv("results/summary_by_condition.csv", index=False)
print("=== 汇总结果（按难度与角色） ===")
print(summary)

# ============ 2. 总体准确率（不分难度） ============
overall_acc = df.groupby("condition")["correct"].mean().reset_index()
print("\n=== 总体准确率（不分难度） ===")
print(overall_acc)

# ============ 3. 按难度分组的准确率 ============
acc_by_diff = df.groupby(["difficulty", "condition"], observed=True)["correct"].mean().reset_index()
print("\n=== 分难度准确率 ===")
print(acc_by_diff)

# ============ 4. 作图 ============

# 准备 pivot 数据
pivot_acc = summary.pivot(index="difficulty", columns="condition", values="accuracy").reindex(DIFF_ORDER)
pivot_lat = summary.pivot(index="difficulty", columns="condition", values="avg_latency").reindex(DIFF_ORDER)
pivot_tok = summary.pivot(index="difficulty", columns="condition", values="avg_completion_tokens").reindex(DIFF_ORDER)

# 图1：准确率对比（三个难度）—— 图例改为左上角
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(pivot_acc))
width = 0.35
ax.bar(x - width/2, pivot_acc["baseline"], width, label="无角色")
ax.bar(x + width/2, pivot_acc["mathematician"], width, label="数学家")
ax.set_xticks(x)
ax.set_xticklabels([DIFF_LABELS[d] for d in pivot_acc.index])
ax.set_ylabel("准确率")
ax.set_title("角色扮演对推理准确率的影响（按难度）")
ax.set_ylim(0, 1.1)
for i, v in enumerate(pivot_acc["baseline"]):
    ax.text(i - width/2, v + 0.02, f"{v:.1%}", ha="center")
for i, v in enumerate(pivot_acc["mathematician"]):
    ax.text(i + width/2, v + 0.02, f"{v:.1%}", ha="center")
ax.legend()
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png", dpi=300)
plt.close()

# 图2：平均延迟对比（三个难度）
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, pivot_lat["baseline"], width, label="无角色")
ax.bar(x + width/2, pivot_lat["mathematician"], width, label="数学家")
ax.set_xticks(x)
ax.set_xticklabels([DIFF_LABELS[d] for d in pivot_lat.index])
ax.set_ylabel("平均延迟（秒）")
ax.set_title("角色扮演对响应时间的影响（按难度）")
for i, v in enumerate(pivot_lat["baseline"]):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center")
for i, v in enumerate(pivot_lat["mathematician"]):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center")
ax.legend()
plt.tight_layout()
plt.savefig("figures/latency_comparison.png", dpi=300)
plt.close()

# 图3：平均输出 token 对比（三个难度）
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, pivot_tok["baseline"], width, label="无角色")
ax.bar(x + width/2, pivot_tok["mathematician"], width, label="数学家")
ax.set_xticks(x)
ax.set_xticklabels([DIFF_LABELS[d] for d in pivot_tok.index])
ax.set_ylabel("平均输出 token 数")
ax.set_title("角色扮演对输出 token 消耗的影响（按难度）")
for i, v in enumerate(pivot_tok["baseline"]):
    ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha="center")
for i, v in enumerate(pivot_tok["mathematician"]):
    ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha="center")
ax.legend()
plt.tight_layout()
plt.savefig("figures/token_comparison.png", dpi=300)
plt.close()

print("\n图表已保存到 figures/：")
print("  - accuracy_comparison.png   （三难度准确率对比）")
print("  - latency_comparison.png    （三难度延迟对比）")
print("  - token_comparison.png      （三难度 token 消耗对比）")
print("汇总表已保存到 results/summary_by_condition.csv")