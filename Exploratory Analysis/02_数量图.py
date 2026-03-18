# -*- coding: utf-8 -*-数量图
"""
EDA: Quarterly text count (2023Q1–2025Q3) for Weibo (Chinese) & Twitter (English)
- Read CN CSV + EN CSV
- Auto-detect time column by parse success rate
- Parse timestamps / common date formats
- Aggregate by quarter (2023Q1–2025Q3), fill missing quarters with 0
- Export:
  * EDA_quarterly_counts_2023Q1_2025Q3.csv
  * EDA_Weibo_Twitter_quarterly_counts_raw.png/.pdf
  * EDA_Weibo_Twitter_quarterly_counts_normalized.png/.pdf
  * EDA_Weibo_quarterly_counts.png/.pdf
  * EDA_Twitter_quarterly_counts.png/.pdf

NOTE:
- Figure titles do NOT include "(2023Q1–2025Q3)".
- Legends use "Weibo (Chinese)" and "Twitter (English)".
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# ========== Paths (edit if needed) ==========
CN_CSV = Path("weibo_ai_2023_2025_final_clean.csv")   # Weibo CN data
EN_CSV = Path("EN_DATA/PREP/prepared_corpus.csv")            # Twitter EN prepared corpus

# ========== Quarter range ==========
START_Q = "2023Q1"
END_Q   = "2025Q3"

# Candidate time columns (CN/EN common)
TIME_CAND_COLS = [
    "时间","发布时间","微博时间","created_at","publish_time","time","date","日期",
    "datetime","created_time","created","timestamp","ts"
]

# Optional: if you KNOW the time column, set it here (else None)
FORCE_CN_TIME_COL = "时间"   # set None if not sure
FORCE_EN_TIME_COL = None     # e.g., "created_at" if you know it

# Output prefix
OUT_PREFIX = "EDA"

# ================== Time parsing ==================
_re_space = re.compile(r"\s+")

def _normalize_time_str(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return x

    # 13/10-digit unix timestamp
    if re.fullmatch(r"\d{13}", x):
        try:
            return pd.to_datetime(int(x), unit="ms").isoformat()
        except Exception:
            pass
    if re.fullmatch(r"\d{10}", x):
        try:
            return pd.to_datetime(int(x), unit="s").isoformat()
        except Exception:
            pass

    # Chinese date markers
    x = x.replace("年", "-").replace("月", "-").replace("日", " ")
    # common separators
    x = x.replace("/", "-").replace(".", "-")
    # unify whitespaces
    x = _re_space.sub(" ", x).strip()
    return x

def parse_time_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).map(_normalize_time_str)
    return pd.to_datetime(s2, errors="coerce", utc=False)

def pick_time_col(df: pd.DataFrame, force_col=None) -> str:
    if force_col and force_col in df.columns:
        return force_col

    best_col, best_ok = None, -1
    for c in TIME_CAND_COLS:
        if c in df.columns:
            ok = parse_time_series(df[c]).notna().sum()
            if ok > best_ok:
                best_ok, best_col = ok, c

    # fallback: brute-force scan (limited)
    if best_col is None:
        cand = list(df.columns)
        obj_cols = [c for c in cand if df[c].dtype == "O"]
        cand2 = obj_cols + [c for c in cand if c not in obj_cols]
        for c in cand2[:30]:
            try:
                ok = parse_time_series(df[c]).notna().sum()
                if ok > best_ok:
                    best_ok, best_col = ok, c
            except Exception:
                continue

    if best_col is None or best_ok <= 0:
        raise ValueError("No parsable time column found. Please set FORCE_*_TIME_COL.")
    return best_col

# ================== Quarter helpers ==================
def quarter_str(dt: pd.Timestamp) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"

def gen_expected_quarters(start_q: str, end_q: str):
    sy, sq = int(start_q[:4]), int(start_q[-1])
    ey, eq = int(end_q[:4]), int(end_q[-1])
    out = []
    y, q = sy, sq
    while True:
        out.append(f"{y}Q{q}")
        if y == ey and q == eq:
            break
        q += 1
        if q == 5:
            q = 1
            y += 1
    return out

EXPECTED_Q = gen_expected_quarters(START_Q, END_Q)

def filter_dt_range(dt: pd.Series, start_q: str, end_q: str) -> pd.Series:
    sy, sq = int(start_q[:4]), int(start_q[-1])
    ey, eq = int(end_q[:4]), int(end_q[-1])
    start_month = (sq - 1) * 3 + 1
    end_month = (eq - 1) * 3 + 3
    start_dt = pd.Timestamp(sy, start_month, 1)
    end_dt = (pd.Timestamp(ey, end_month, 1) + pd.offsets.MonthEnd(0))
    return (dt >= start_dt) & (dt <= end_dt)

def quarterly_counts(df: pd.DataFrame, time_col: str, label: str):
    dt = parse_time_series(df[time_col])
    ok = dt.notna()
    df2 = df.loc[ok].copy()
    df2["_dt"] = dt.loc[ok]

    m = filter_dt_range(df2["_dt"], START_Q, END_Q)
    df2 = df2.loc[m].copy()

    df2["quarter"] = df2["_dt"].map(quarter_str)
    counts_dict = df2["quarter"].value_counts().to_dict()
    counts = pd.Series([int(counts_dict.get(q, 0)) for q in EXPECTED_Q],
                       index=EXPECTED_Q, name=label)
    return counts, len(df2), ok.sum(), len(df)

# ================== Plotting ==================
def save_line_plot(x_labels, y_map, title, ylabel, out_png, out_pdf, legend_loc="best"):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))

    for lab, y in y_map.items():
        ax.plot(x_labels, list(y), marker="o", linewidth=2.2, markersize=8, label=lab)

    ax.set_xlabel("Quarter", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20, fontweight="bold")

    ax.tick_params(axis='x', labelsize=20, rotation=35, length=6, width=1.2, direction='out')
    ax.tick_params(axis='y', labelsize=15, length=6, width=1.2, direction='out')

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc=legend_loc, fontsize=11, frameon=True)

    fig.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def main():
    assert CN_CSV.exists(), f"Missing CN_CSV: {CN_CSV}"
    assert EN_CSV.exists(), f"Missing EN_CSV: {EN_CSV}"

    df_cn = pd.read_csv(CN_CSV, encoding="utf-8-sig")
    df_en = pd.read_csv(EN_CSV, encoding="utf-8-sig")

    cn_time_col = pick_time_col(df_cn, FORCE_CN_TIME_COL)
    en_time_col = pick_time_col(df_en, FORCE_EN_TIME_COL)

    print(f"[CN] time col = {cn_time_col}")
    print(f"[EN] time col = {en_time_col}")

    cn_counts, cn_inrange_n, cn_parse_ok, cn_total = quarterly_counts(df_cn, cn_time_col, "Weibo (Chinese)")
    en_counts, en_inrange_n, en_parse_ok, en_total = quarterly_counts(df_en, en_time_col, "Twitter (English)")

    print(f"[Weibo]   total={cn_total} | time-parsable={cn_parse_ok} | in-range={cn_inrange_n}")
    print(f"[Twitter] total={en_total} | time-parsable={en_parse_ok} | in-range={en_inrange_n}")

    # Export table
    out_df = pd.DataFrame({
        "quarter": EXPECTED_Q,
        "weibo_chinese_count": cn_counts.values,
        "twitter_english_count": en_counts.values,
    })
    out_csv = f"{OUT_PREFIX}_quarterly_counts_{START_Q}_{END_Q}.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[save] {out_csv}")

    # 1) Combined raw
    save_line_plot(
        EXPECTED_Q,
        {"Weibo (Chinese)": cn_counts, "Twitter (English)": en_counts},
        title="Quarterly Distribution of Text Count",
        ylabel="Number of Texts",
        out_png=f"{OUT_PREFIX}_Weibo_Twitter_quarterly_counts_raw.png",
        out_pdf=f"{OUT_PREFIX}_Weibo_Twitter_quarterly_counts_raw.pdf",
        legend_loc="best"
    )
    print(f"[save] {OUT_PREFIX}_Weibo_Twitter_quarterly_counts_raw.png/.pdf")

    # 2) Combined normalized (0–1) for trend comparison
    cn_max = max(int(cn_counts.max()), 1)
    en_max = max(int(en_counts.max()), 1)
    cn_norm = cn_counts / cn_max
    en_norm = en_counts / en_max

    save_line_plot(
        EXPECTED_Q,
        {"Weibo (Chinese) - normalized": cn_norm, "Twitter (English) - normalized": en_norm},
        title="Quarterly Trend Comparison (Normalized)",
        ylabel="Normalized Count (0–1)",
        out_png=f"{OUT_PREFIX}_Weibo_Twitter_quarterly_counts_normalized.png",
        out_pdf=f"{OUT_PREFIX}_Weibo_Twitter_quarterly_counts_normalized.pdf",
        legend_loc="best"
    )
    print(f"[save] {OUT_PREFIX}_Weibo_Twitter_quarterly_counts_normalized.png/.pdf")

    # 3) Separate plots (raw)
    save_line_plot(
        EXPECTED_Q,
        {"Weibo (Chinese)": cn_counts},
        title="Weibo Corpus: Quarterly Text Count",
        ylabel="Number of Texts",
        out_png=f"{OUT_PREFIX}_Weibo_quarterly_counts.png",
        out_pdf=f"{OUT_PREFIX}_Weibo_quarterly_counts.pdf",
        legend_loc="best"
    )
    print(f"[save] {OUT_PREFIX}_Weibo_quarterly_counts.png/.pdf")

    save_line_plot(
        EXPECTED_Q,
        {"Twitter (English)": en_counts},
        title="Twitter Corpus: Quarterly Text Count",
        ylabel="Number of Texts",
        out_png=f"{OUT_PREFIX}_Twitter_quarterly_counts.png",
        out_pdf=f"{OUT_PREFIX}_Twitter_quarterly_counts.pdf",
        legend_loc="best"
    )
    print(f"[save] {OUT_PREFIX}_Twitter_quarterly_counts.png/.pdf")

    print("\n[done] Exported figures with legends: Weibo (Chinese) & Twitter (English) ✅")

if __name__ == "__main__":
    main()
