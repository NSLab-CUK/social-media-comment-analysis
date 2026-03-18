# -*- coding: utf-8 -*- k

import os, re, ast, math, warnings
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")

# ========= 配置 =========
OUTPUT_PREFIX = "10.14工作"
SEED = 42
K_RANGE = list(range(3, 13))         # 3..12
START_YEAR, START_Q = 2023, 1
END_YEAR, END_Q = 2025, 3

# 老师风格（True 更接近老师）
TEACHER_MODE = False
if TEACHER_MODE:
    PASSES = 30; ITERATIONS = 100; ALPHA = 'symmetric'; ETA = None
    NO_BELOW_MIN = 1; NO_ABOVE = 1.0
else:
    PASSES = 15; ITERATIONS = 300; ALPHA = 'asymmetric'; ETA = 'auto'
    NO_BELOW_MIN = 5; NO_ABOVE = 0.5

KEEP_N = 200000
CHUNKSIZE = 2000
MIN_TOKENS_PER_DOC = 2        # 如果样本仍偏少，可临时降到 1 观察
MIN_DOCS_PER_SLICE = 20

RAW_INPUT_CSV = Path("weibo_ai_2023_2025_final_clean.csv")
TOKENS_CSV = Path("weibo_ai_tokens_2023_2025.csv")

TEXT_CAND = ["内容", "文本", "text", "微博正文", "content", "full_text"]
FORCE_DT_COL: Optional[str] = "时间"   # 你的时间列

# ========= 清洗文本 & 时间解析 =========
_re_url   = re.compile(r'https?://\S+|www\.\S+')
_re_at    = re.compile(r'@[\w\-\u4e00-\u9fff]+')
_re_topic = re.compile(r'#([^#]+)#')
_re_space = re.compile(r'\s+')
_re_keep  = re.compile(r'[A-Za-z0-9\u4e00-\u9fa5]+')

def clean_text_func(s: str) -> str:
    if not isinstance(s, str): return ""
    s = _re_url.sub(" ", s)
    s = _re_at.sub(" ", s)
    s = _re_topic.sub(" ", s)
    s = _re_space.sub(" ", s).strip()
    parts = _re_keep.findall(s)
    return " ".join(parts)

def find_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CAND:
        if c in df.columns: return c
    obj = [c for c in df.columns if df[c].dtype == 'O']
    if obj:
        lens = {c: df[c].astype(str).str.len().mean() for c in obj}
        return max(lens, key=lens.get)
    raise ValueError("未找到文本列")

def _normalize_time_str(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.strip()
    if not s: return s
    # 13/10位时间戳
    if re.fullmatch(r"\d{13}", s):
        try: return pd.to_datetime(int(s), unit="ms").isoformat()
        except: pass
    if re.fullmatch(r"\d{10}", s):
        try: return pd.to_datetime(int(s), unit="s").isoformat()
        except: pass
    # 中文“年/月/日” + 常见分隔
    s = s.replace("年","-").replace("月","-").replace("日"," ")
    s = s.replace("/", "-").replace(".", "-")
    # 统一空白
    s = re.sub(r"\s+"," ", s).strip()
    return s

COMMON_FORMATS = [
    "%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S%Z","%Y-%m-%dT%H:%M:%S",
]

def parse_time_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).map(_normalize_time_str)
    dt = pd.to_datetime(s2, errors="coerce", utc=False, infer_datetime_format=True)
    if dt.notna().sum() == 0:
        for fmt in COMMON_FORMATS:
            dt2 = pd.to_datetime(s2, format=fmt, errors="coerce")
            if dt2.notna().sum() > 0:
                return dt2
    return dt

def gen_expected_quarters(y1: int, q1: int, y2: int, q2: int) -> list:
    out=[]; y,q=y1,q1
    while True:
        out.append(f"{y}Q{q}")
        if y==y2 and q==q2: break
        q+=1
        if q==5: q=1; y+=1
    return out

def quarter_label(dt: pd.Timestamp) -> str:
    return f"{dt.year}Q{(dt.month-1)//3+1}"

# ========= LDA 工具 =========
def build_dtm(docs: List[List[str]]):
    dictionary = corpora.Dictionary(docs)
    auto_no_below = max(NO_BELOW_MIN, int(len(docs)*0.001))  # 覆盖≥0.1%
    dictionary.filter_extremes(no_below=auto_no_below, no_above=NO_ABOVE, keep_n=KEEP_N)
    corpus = [dictionary.doc2bow(text) for text in docs]
    return dictionary, corpus

def train_and_score(dictionary, corpus, docs, k: int):
    lda = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=k,
        passes=PASSES, iterations=ITERATIONS, alpha=ALPHA, eta=ETA,
        random_state=SEED, chunksize=CHUNKSIZE, eval_every=None,
        minimum_probability=0.0, per_word_topics=False,
    )
    log_perp = lda.log_perplexity(corpus)
    perp = math.exp(-log_perp)  # 越低越好
    cm = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence="c_v",
                        processes=max(1, (os.cpu_count() or 2) - 1))
    c_v = cm.get_coherence()    # 越高越好
    return c_v, log_perp, perp

def plot_xy(x, y, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(7.6, 4.6))
    plt.rcParams["font.sans-serif"]=["SimHei","Arial Unicode MS","DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"]=False
    plt.plot(x, y, marker="o"); plt.xticks(x)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=0.3, linestyle="--"); plt.tight_layout()
    plt.savefig(out_png, dpi=180); plt.close()
    print(f"[save] {out_png} ✅")

# ========= 主流程 =========
def main():
    assert TOKENS_CSV.exists(), "缺少 weibo_ai_tokens_2023_2025.csv"
    assert RAW_INPUT_CSV.exists(), "缺少 weibo_ai_2023_2025_final_clean.csv"

    # 1) 读取 tokens 表（主表）
    df_tok = pd.read_csv(TOKENS_CSV, encoding="utf-8-sig")
    assert "clean_text" in df_tok.columns and "tokens" in df_tok.columns, "tokens CSV 必须含 clean_text/tokens"
    toks=[]
    for x in df_tok["tokens"].astype(str):
        try:
            arr = ast.literal_eval(x)
            arr = [str(w).strip() for w in arr if str(w).strip()]
        except: arr=[]
        toks.append(arr)
    df_tok["tokens"]=toks
    df_tok["tok_len"]=df_tok["tokens"].apply(len)

    # 2) 读取原始CSV，生成 clean_text 并解析“时间”列
    df_raw = pd.read_csv(RAW_INPUT_CSV, encoding="utf-8-sig")
    text_col = find_text_col(df_raw)
    df_raw["_clean_text"] = df_raw[text_col].astype(str).map(clean_text_func)

    time_col = FORCE_DT_COL if (FORCE_DT_COL and FORCE_DT_COL in df_raw.columns) else FORCE_DT_COL
    if time_col is None or time_col not in df_raw.columns:
        # 兜底：自动找一个最像时间的列（可解析率最高）
        best_name, best_rate, best_series = None, -1.0, None
        for c in df_raw.columns:
            dt = parse_time_series(df_raw[c])
            rate = dt.notna().mean()
            if rate>best_rate:
                best_name, best_rate, best_series = c, rate, dt
        time_col = best_name
        df_raw["_dt"] = best_series
        print(f"[time] 自动选择时间列：{time_col} | 可解析率={best_rate:.2%}")
    else:
        df_raw["_dt"] = parse_time_series(df_raw[time_col])
        print(f"[time] 使用时间列：{time_col} | 可解析：{df_raw['_dt'].notna().sum()}/{len(df_raw)}")

    # 2.1 对原始表按 clean_text 去重（保留第一条），模拟你的清洗逻辑
    df_map = df_raw.dropna(subset=["_clean_text"]).sort_index() \
                   .drop_duplicates(subset=["_clean_text"], keep="first")[["_clean_text","_dt"]]

    # 3) 按 clean_text 合并时间到 tokens 表
    df = df_tok.merge(df_map, left_on="clean_text", right_on="_clean_text", how="left")
    matched = df["_dt"].notna().sum()
    print(f"[align] 按 clean_text 成功匹配时间：{matched}/{len(df)}")
    # 报告未匹配样本（便于排查）
    if matched < len(df):
        df[df["_dt"].isna()][["clean_text"]].head(50).to_csv(f"{OUTPUT_PREFIX}_unmatched_clean_text_samples.csv",
                                                             index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_unmatched_clean_text_samples.csv （前50条未匹配 clean_text）")

    # 4) 过滤：有时间 & token 长度阈值
    df = df[df["_dt"].notna()].copy()
    df["_dt"] = pd.to_datetime(df["_dt"], errors="coerce")
    before_len = len(df)
    df = df[df["tok_len"] >= MIN_TOKENS_PER_DOC].reset_index(drop=True)
    print(f"[filter] token≥{MIN_TOKENS_PER_DOC}：{len(df)}/{before_len}")

    # 5) 季度切片
    df["quarter"] = df["_dt"].apply(quarter_label)
    expected = gen_expected_quarters(START_YEAR, START_Q, END_YEAR, END_Q)
    present = sorted(df["quarter"].unique())
    quarters = [q for q in expected if q in present] or present
    per_q_counts = df["quarter"].value_counts().to_dict()
    print(f"[plan] 季度序列：{quarters}")
    print(f"[diag] 每季样本数：{per_q_counts}")

    # 掉点报告
    drop_rows = [
        {"stage":"tokens_rows", "rows": len(df_tok)},
        {"stage":"matched_time", "rows": matched},
        {"stage":f"toklen>= {MIN_TOKENS_PER_DOC}", "rows": len(df)},
    ] + [{"stage": f"quarter_{q}", "rows": int(per_q_counts.get(q,0))} for q in quarters]
    pd.DataFrame(drop_rows).to_csv(f"{OUTPUT_PREFIX}_drop_report.csv", index=False, encoding="utf-8-sig")
    print(f"[save] {OUTPUT_PREFIX}_drop_report.csv ✅")

    # 6) 逐季度扫 K
    summary_rows = []
    all_k_rows = []

    for qlab in quarters:
        sub = df[df["quarter"]==qlab]
        n = len(sub)
        print(f"\n[slice] {qlab}: 文档数={n}")
        if n < MIN_DOCS_PER_SLICE: print(f"[warn] {qlab} 文档较少，结果可能不稳定。")
        docs = sub["tokens"].tolist()
        dictionary, corpus = build_dtm(docs)
        if len(dictionary)==0 or len(corpus)==0:
            print(f"[skip] {qlab} 词典/语料为空，跳过。"); continue

        recs=[]; best_cv=-1.0; best_k_cv=None; best_perp=float("inf"); best_k_perp=None
        for k in K_RANGE:
            try:
                c_v, log_perp, perp = train_and_score(dictionary, corpus, docs, k)
            except Exception as e:
                print(f"[err] {qlab} K={k} 失败：{e}"); continue
            rec = {"quarter": qlab, "docs": n, "k": k,
                   "coherence_c_v": c_v, "log_perplexity": log_perp, "perplexity": perp}
            recs.append(rec); all_k_rows.append(rec)
            if c_v>best_cv: best_cv, best_k_cv = c_v, k
            if perp<best_perp: best_perp, best_k_perp = perp, k
            print(f"  K={k:>2d} | c_v={c_v:.4f} | log_perp={log_perp:.4f} | perp={perp:.2f}")

        if not recs: continue
        dfq = pd.DataFrame(recs).sort_values("k")
        dfq.to_csv(f"{OUTPUT_PREFIX}_Q_{qlab}_k_selection.csv", index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_Q_{qlab}_k_selection.csv ✅")

        plot_xy(dfq["k"].tolist(), dfq["coherence_c_v"].tolist(),
                "主题数 K", "一致性 c_v（越高越好）",
                f"{qlab}：c_v 一致性 vs K", f"{OUTPUT_PREFIX}_Q_{qlab}_k_vs_coherence.png")
        plot_xy(dfq["k"].tolist(), dfq["perplexity"].tolist(),
                "主题数 K", "Perplexity（越低越好）",
                f"{qlab}：困惑度 Perplexity vs K", f"{OUTPUT_PREFIX}_Q_{qlab}_k_vs_perplexity.png")

        summary_rows.append({
            "quarter": qlab, "docs": n,
            "K_by_c_v": best_k_cv, "c_v_max": round(best_cv,4),
            "K_by_perplexity": best_k_perp, "perplexity_min": round(best_perp,2),
        })
        print(f"[choose] {qlab} | K_by_c_v={best_k_cv} (c_v={best_cv:.4f}) | "
              f"K_by_perplexity={best_k_perp} (perp={best_perp:.2f})")

    # 7) 汇总 & 全局最佳K
    df_sum = pd.DataFrame(summary_rows).sort_values("quarter")
    if not df_sum.empty:
        df_sum.to_csv(f"{OUTPUT_PREFIX}_quarterly_k_best.csv", index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_quarterly_k_best.csv ✅")

    df_allk = pd.DataFrame(all_k_rows).sort_values(["quarter","k"])
    if not df_allk.empty:
        df_allk.to_csv(f"{OUTPUT_PREFIX}_all_k_selection.csv", index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_all_k_selection.csv ✅")

        g = df_allk.groupby("k").agg(
            mean_c_v=("coherence_c_v","mean"),
            std_c_v=("coherence_c_v","std"),
            mean_perp=("perplexity","mean"),
            std_perp=("perplexity","std"),
            n_quarters=("quarter","nunique")
        ).reset_index().sort_values("k")
        g.to_csv(f"{OUTPUT_PREFIX}_global_k_selection.csv", index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_global_k_selection.csv ✅")

        # 全局最佳：一致性/困惑度
        k_cv   = int(g.loc[g["mean_c_v"].idxmax(),"k"])
        k_perp = int(g.loc[g["mean_perp"].idxmin(),"k"])

        # 打平手口径
        eps_cv   = 0.005
        max_cv   = g["mean_c_v"].max()
        coh_cand = g[g["mean_c_v"] >= max_cv - eps_cv]
        k_coh_first = int(coh_cand.sort_values(["mean_perp","k"]).iloc[0]["k"])

        eps_perp = 0.005 * float(g["mean_perp"].mean() or 1.0)
        min_perp = g["mean_perp"].min()
        perp_cand = g[g["mean_perp"] <= min_perp + eps_perp]
        k_perp_first = int(perp_cand.sort_values(["mean_c_v","k"], ascending=[False,True]).iloc[0]["k"])

        final_rows = [
            {"method":"global_best_k_by_cv","k":k_cv,"note":"跨季度平均 c_v 最大"},
            {"method":"global_best_k_by_perp","k":k_perp,"note":"跨季度平均 perplexity 最小"},
            {"method":"final_best_k_coh_first","k":k_coh_first,"note":"一致性优先，困惑度打平手"},
            {"method":"final_best_k_perp_first","k":k_perp_first,"note":"困惑度优先，一致性打平手"},
        ]
        pd.DataFrame(final_rows).to_csv(f"{OUTPUT_PREFIX}_final_bestK.csv", index=False, encoding="utf-8-sig")
        print(f"[save] {OUTPUT_PREFIX}_final_bestK.csv ✅")

    # 8) 汇总到一个 Excel
    try:
        with pd.ExcelWriter(f"{OUTPUT_PREFIX}_all_results.xlsx", engine="openpyxl") as writer:
            if not df_sum.empty:
                df_sum.to_excel(writer, sheet_name="summary_bestK_by_quarter", index=False)
            if not df_allk.empty:
                df_allk.to_excel(writer, sheet_name="all_k_selection", index=False)
                g.to_excel(writer, sheet_name="global_selection", index=False)
                pd.DataFrame(final_rows).to_excel(writer, sheet_name="final_bestK", index=False)
            pd.DataFrame(sorted(per_q_counts.items()), columns=["quarter","docs"])\
              .to_excel(writer, sheet_name="counts_by_quarter", index=False)
            pd.DataFrame(drop_rows).to_excel(writer, sheet_name="drop_report", index=False)
        print(f"[save] {OUTPUT_PREFIX}_all_results.xlsx ✅")
    except Exception as e:
        print(f"[warn] 写入 Excel 失败：{e}；已保留 CSV 备份。")

    print("\n[done] 季度切片评估 + 全局最佳K 完成 ✅")

if __name__ == "__main__":
    main()
