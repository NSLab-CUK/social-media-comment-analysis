# ====== Plot: global mean C_v vs K + quarterly C_v vs K ======画k图
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---- Basic display settings ----
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

GLOBAL_CSV = "global_k_selection.csv"
ALLK_CSV   = "all_k_selection.csv"

# ============ 1) Global: mean C_v across quarters vs K ============
g = pd.read_csv(GLOBAL_CSV, encoding="utf-8-sig")

plt.figure(figsize=(7.6, 4.6))
plt.plot(g["k"], g["mean_c_v"], marker="o", linewidth=2)

plt.xlabel("Number of Topics ($K$)", fontsize=20)
plt.ylabel("Mean $C_v$ Across Quarters", fontsize=20)
plt.title(r"Global Mean $C_v$ vs $K$", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("plot_global_mean_cv_vs_k.png", dpi=300, bbox_inches="tight")
plt.show()

print("[save] plot_global_mean_cv_vs_k.png ✅")


# ============ 2) Quarterly: one curve per quarter ============
df_all = pd.read_csv(ALLK_CSV, encoding="utf-8-sig")

# Ensure sorting
df_all = df_all.sort_values(["quarter", "k"]).reset_index(drop=True)

plt.figure(figsize=(9.2, 5.4))

# Plot one line for each quarter
for q, sub in df_all.groupby("quarter"):
    plt.plot(sub["k"], sub["coherence_c_v"], marker="o", linewidth=1.5, label=q)

plt.xlabel("Number of Topics (K)")
plt.ylabel("C_v Score")
plt.title("Quarterly C_v vs. K")

plt.grid(alpha=0.3, linestyle="--")

# Put legend on the right
plt.legend(title="Quarter", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
plt.tight_layout()
plt.savefig("10.14工作_plot_quarterly_cv_vs_k.png", dpi=300, bbox_inches="tight")
plt.show()

print("[save] 10.14工作_plot_quarterly_cv_vs_k.png ✅")