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

# 保证排序
DIFF_ORDER = ["simple", "medium", "hard"]
COND_ORDER = ["baseline", "mathematician", "efficient_math", "careless_student"]
COND_LABELS = {
    "baseline": "无角色",
    "mathematician": "数学家",
    "efficient_math": "高效数学家",
    "careless_student": "粗心学生"
}
df["difficulty"] = pd.Categorical(df["difficulty"], categories=DIFF_ORDER, ordered=True)

# ============ 1. 汇总统计 ============
summary = df.groupby(["difficulty", "condition"], observed=True).agg(
    accuracy=("correct", "mean"),
    avg_latency=("latency", "mean"),
    avg_tokens=("completion_tokens", "mean"),
    count=("correct", "count"),
).reset_index()

# 计算效率指标：每秒答对题数（Accuracy / Latency）
summary["efficiency_speed"] = summary["accuracy"] / summary["avg_latency"]

summary.to_csv("results/summary_by_condition.csv", index=False)
print("=== 汇总结果 ===")
print(summary)

# ============ 2. 绘图与可视化 ============

# 图1：准确率对比（柱状图）
fig, ax = plt.subplots(figsize=(10, 5))
pivot_acc = summary.pivot(index="difficulty", columns="condition", values="accuracy").reindex(columns=COND_ORDER)
x = np.arange(len(pivot_acc))
width = 0.2
for i, cond in enumerate(COND_ORDER):
    if cond in pivot_acc.columns:
        ax.bar(x + (i - 1.5) * width, pivot_acc[cond], width, label=COND_LABELS[cond])
ax.set_xticks(x)
ax.set_xticklabels([{"simple":"简单题","medium":"中等题","hard":"困难题"}[d] for d in pivot_acc.index])
ax.set_ylabel("准确率")
ax.set_title("不同角色设定对推理准确率的影响")
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png", dpi=300)
plt.close()

# 图2：延迟与Token的箱线图（观察数据分布）
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 延迟箱线图
df.boxplot(column="latency", by="condition", ax=axes[0], grid=False)
axes[0].set_title("不同角色的响应延迟分布")
axes[0].set_ylabel("延迟 (秒)")
axes[0].set_xlabel("角色")
axes[0].set_xticklabels([COND_LABELS.get(l.get_text(), l.get_text()) for l in axes[0].get_xticklabels()])

# Token箱线图
df.boxplot(column="completion_tokens", by="condition", ax=axes[1], grid=False)
axes[1].set_title("不同角色的输出Token分布")
axes[1].set_ylabel("输出 Token 数")
axes[1].set_xlabel("角色")
axes[1].set_xticklabels([COND_LABELS.get(l.get_text(), l.get_text()) for l in axes[1].get_xticklabels()])
plt.suptitle("") # 去掉默认的 suptitle
plt.tight_layout()
plt.savefig("figures/boxplots_performance.png", dpi=300)
plt.close()

# 图3：不同角色的综合效率（每秒答对题数）
fig, ax = plt.subplots(figsize=(10, 5))
pivot_eff = summary.pivot(index="difficulty", columns="condition", values="efficiency_speed").reindex(columns=COND_ORDER)
for i, cond in enumerate(COND_ORDER):
    if cond in pivot_eff.columns:
        ax.bar(x + (i - 1.5) * width, pivot_eff[cond], width, label=COND_LABELS[cond])
ax.set_xticks(x)
ax.set_xticklabels([{"simple":"简单题","medium":"中等题","hard":"困难题"}[d] for d in pivot_eff.index])
ax.set_ylabel("综合效率 (答对题数 / 秒)")
ax.set_title("角色设定对系统吞吐效率的影响")
ax.legend()
plt.tight_layout()
plt.savefig("figures/efficiency_comparison.png", dpi=300)
plt.close()

print("\n图表已保存到 figures/ 目录：")
print("  1. accuracy_comparison.png   (准确率对比)")
print("  2. boxplots_performance.png  (性能分布箱线图)")
print("  3. efficiency_comparison.png (综合吞吐效率)")
print("汇总表已保存到 results/summary_by_condition.csv")