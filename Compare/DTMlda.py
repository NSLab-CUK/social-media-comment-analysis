# compare_cn_dtm_vs_quarterly_lda_teacher_style_v2_matched_first.py
# -*- coding: utf-8 -*-对比lda dtm

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from sentence_transformers import SentenceTransformer
import torch


# ===================== 1) 配置区（只改这里） =====================
BASE_DIR = Path(".")

DTM_TOPIC_DIR = BASE_DIR / "CN_DTM_TopicWords_Out"   # dtm_topics_2023Q1.csv ...
LDA_TOPIC_DIR = BASE_DIR / "LDA_Out"                # lda_topics_2023Q1.csv ...

OUT_DIR = BASE_DIR / "Compare_CN_LDA_DTM_Out"

EMBED_MODEL_NAME = "BAAI/bge-m3"
TOPN_FOR_TEXT = 30

CMAP = "YlGnBu"
THEME = "white"
FIGSIZE = (10, 10)
DPI = 220
ANNOT_WHITE_TH = 0.65

# ✅ 新增：center 后再做一次 L2 normalize（通常会更“尖锐”一点）
RENORM_AFTER_CENTER = True

DOMAIN_STOPWORDS = {
    "人工智能", "ai", "a.i", "智能", "技术", "科技", "发展", "未来", "研究", "算法", "模型", "数据",
    "系统", "应用", "能力", "训练", "学习", "网络", "平台", "产业", "行业", "公司", "企业",
    "中国", "美国", "世界", "人类", "社会", "问题", "可能", "现在", "如果", "就是", "这个", "那个",
    "gpt", "chatgpt", "openai", "大模型", "语言模型"
}

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 2) 工具函数 =====================
def detect_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _safe_word(w: str):
    w = str(w).strip()
    if not w or w.lower() == "nan":
        return None
    if w.lower() in DOMAIN_STOPWORDS:
        return None
    return w


def load_topics_any_format(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = set(df.columns)

    # long
    if {"topic", "rank", "word"}.issubset(cols):
        out = df[["topic", "rank", "word"]].copy()
        out["weight"] = df["weight"] if "weight" in df.columns else 1.0
        out["topic"] = out["topic"].astype(int)
        out["rank"] = out["rank"].astype(int)
        return out.sort_values(["topic", "rank"]).reset_index(drop=True)

    # wide -> long
    if "topic" not in cols:
        raise ValueError(f"{path.name} 缺少 topic 列，无法识别格式。")

    word_cols = [c for c in df.columns if re.fullmatch(r"word_\d+", str(c))]
    if not word_cols:
        raise ValueError(f"{path.name} 看起来像 wide，但没找到 word_1.. 列。")

    word_cols = sorted(word_cols, key=lambda x: int(x.split("_")[1]))
    rows = []
    for _, r in df.iterrows():
        tid = int(r["topic"])
        for i, wc in enumerate(word_cols, 1):
            w = r.get(wc, "")
            if str(w).strip() == "" or str(w).lower() == "nan":
                continue
            wt = r.get(f"weight_{i}", 1.0)
            try:
                wt = float(wt)
            except Exception:
                wt = 1.0
            rows.append({"topic": tid, "rank": i, "word": str(w), "weight": wt})

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["topic", "rank", "word", "weight"])
    return out.sort_values(["topic", "rank"]).reset_index(drop=True)


def build_topic_text_from_df(df: pd.DataFrame, topic_id: int, topn: int = TOPN_FOR_TEXT) -> str:
    sub = df[df["topic"].astype(int) == int(topic_id)].sort_values("rank").head(topn)
    words = [str(x) for x in sub["word"].tolist()]
    filtered = [x for x in (_safe_word(w) for w in words) if x is not None]
    if not filtered:
        filtered = [w for w in words if str(w).strip() and str(w).lower() != "nan"]
    return " ".join(filtered)


def list_common_quarters(dtm_dir: Path, lda_dir: Path):
    dtm_q = set()
    for p in dtm_dir.glob("dtm_topics_*.csv"):
        m = re.match(r"dtm_topics_(\d{4}Q[1-4])\.csv$", p.name)
        if m:
            dtm_q.add(m.group(1))

    lda_q = set()
    for p in lda_dir.glob("lda_topics_*.csv"):
        m = re.match(r"lda_topics_(\d{4}Q[1-4])\.csv$", p.name)
        if m:
            lda_q.add(m.group(1))

    qs = sorted(list(dtm_q & lda_q), key=lambda x: (int(x[:4]), int(x[-1])))
    return qs


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def plot_heatmap_like_teacher(mat, title, out_png, row_labels, col_labels, xname="LDA Topics"):
    assert sns is not None, "需要 seaborn 才能画 teacher-style 热力图。"

    sns.set_theme(style=THEME)
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    ax = sns.heatmap(
        mat,
        annot=True, fmt=".2f", annot_kws={"size": 20},
        cmap=CMAP, vmin=0.0, vmax=1.0,
        xticklabels=col_labels, yticklabels=row_labels,
        square=True,
        cbar_kws={"label": "Normalized Similarity Score (0-1)"}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Normalized Similarity Score (0-1)", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            continue
        text.set_color("white" if val >= ANNOT_WHITE_TH else "black")

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xname, fontsize=20)
    ax.set_ylabel("DTM Topics", fontsize=20)
    plt.xticks(rotation=35, ha="right", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print("✅ saved figure:", out_png.resolve())


# ===================== 3) 主流程 =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert DTM_TOPIC_DIR.exists(), f"找不到 DTM_TOPIC_DIR：{DTM_TOPIC_DIR.resolve()}"
    assert LDA_TOPIC_DIR.exists(), f"找不到 LDA_TOPIC_DIR：{LDA_TOPIC_DIR.resolve()}"

    quarters = list_common_quarters(DTM_TOPIC_DIR, LDA_TOPIC_DIR)
    if not quarters:
        raise RuntimeError("没有可对比季度：请检查 dtm_topics_<Q>.csv 与 lda_topics_<Q>.csv。")
    print("✅ quarters:", quarters)

    device = detect_device()
    print(f"Loading SentenceTransformer: {EMBED_MODEL_NAME} | device={device}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    # ===== 先编码所有季度，用于 global mean centering =====
    quarter_records = []
    all_embs = []

    for q in quarters:
        dtm_p = DTM_TOPIC_DIR / f"dtm_topics_{q}.csv"
        lda_p = LDA_TOPIC_DIR / f"lda_topics_{q}.csv"

        dtm_df = load_topics_any_format(dtm_p)
        lda_df = load_topics_any_format(lda_p)

        dtm_ids = sorted(dtm_df["topic"].astype(int).unique().tolist())
        lda_ids = sorted(lda_df["topic"].astype(int).unique().tolist())

        dtm_texts = [build_topic_text_from_df(dtm_df, t, topn=TOPN_FOR_TEXT) for t in dtm_ids]
        lda_texts = [build_topic_text_from_df(lda_df, t, topn=TOPN_FOR_TEXT) for t in lda_ids]

        dtm_emb = model.encode(dtm_texts, normalize_embeddings=False)
        lda_emb = model.encode(lda_texts, normalize_embeddings=False)

        quarter_records.append((q, dtm_ids, lda_ids, dtm_emb, lda_emb))
        all_embs.append(dtm_emb)
        all_embs.append(lda_emb)

    global_mean_vec = np.mean(np.vstack(all_embs), axis=0)

    # ===== A) 旧口径：先平均再匹配（保留做对照）=====
    sims_unaligned = []
    dtm_ids_ref, lda_ids_ref = None, None
    for q, dtm_ids, lda_ids, dtm_emb, lda_emb in quarter_records:
        if dtm_ids_ref is None: dtm_ids_ref = dtm_ids
        if lda_ids_ref is None: lda_ids_ref = lda_ids

        dtm_c = dtm_emb - global_mean_vec
        lda_c = lda_emb - global_mean_vec
        if RENORM_AFTER_CENTER:
            dtm_c = l2_normalize_rows(dtm_c)
            lda_c = l2_normalize_rows(lda_c)

        sims_unaligned.append(cosine_similarity(dtm_c, lda_c))

    mean_sim = np.stack(sims_unaligned, axis=0).mean(axis=0)
    mn, mx = float(mean_sim.min()), float(mean_sim.max())
    mean_sim_norm = (mean_sim - mn) / (mx - mn + 1e-12)

    pd.DataFrame(mean_sim).to_csv(OUT_DIR / "mean_sim_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_sim_norm).to_csv(OUT_DIR / "mean_sim_norm01.csv", index=False, encoding="utf-8-sig")

    row_labels = [f"DTM {t}" for t in dtm_ids_ref]
    col_labels = [f"LDA {t}" for t in lda_ids_ref]

    plot_heatmap_like_teacher(
        mean_sim_norm,
        title="Topic Correspondence: Dynamic DTM vs Quarterly LDA\n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels,
        xname="LDA Topics"
    )

    # ===== B) 新口径：每季度先匹配，再平均（更理想）=====
    assert linear_sum_assignment is not None, "需要 scipy 才能做 matched-first。请 pip install scipy"

    sims_matched_each_q = []
    lda_orders_each_q = []

    for q, dtm_ids, lda_ids, dtm_emb, lda_emb in quarter_records:
        dtm_c = dtm_emb - global_mean_vec
        lda_c = lda_emb - global_mean_vec
        if RENORM_AFTER_CENTER:
            dtm_c = l2_normalize_rows(dtm_c)
            lda_c = l2_normalize_rows(lda_c)

        sim = cosine_similarity(dtm_c, lda_c)  # raw cosine (after centering)

        # 每季度匹配：最大化 sim => 最小化 -sim
        r_ind, c_ind = linear_sum_assignment(-sim)

        # 这里 r_ind 通常就是 [0..K-1]，但不强求
        sim_m = sim[r_ind][:, c_ind]

        # 记录该季度 LDA 列重排顺序，方便你后续写论文
        lda_orders_each_q.append((q, c_ind.tolist()))
        sims_matched_each_q.append(sim_m)

    mean_sim_matched = np.stack(sims_matched_each_q, axis=0).mean(axis=0)

    mn2, mx2 = float(mean_sim_matched.min()), float(mean_sim_matched.max())
    mean_sim_matched_norm = (mean_sim_matched - mn2) / (mx2 - mn2 + 1e-12)

    pd.DataFrame(mean_sim_matched).to_csv(OUT_DIR / "mean_sim_matched_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_sim_matched_norm).to_csv(OUT_DIR / "mean_sim_matched_norm01.csv", index=False, encoding="utf-8-sig")

    # matched-first 的最终图：列标签用“被重排后的 LDA”
    # 我们用“全期最常见的列排列”来命名（否则每季度不一样）
    # 简单策略：用第一个季度的 c_ind 作为展示顺序
    show_order = lda_orders_each_q[0][1]
    col_labels_matched = [f"LDA {lda_ids_ref[j]}" for j in show_order]

    plot_heatmap_like_teacher(
        mean_sim_matched_norm,
        title="Topic Correspondence: Dynamic DTM vs Quarterly LDA \n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_matched_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels_matched,
        xname="LDA Topics"
    )

    # 把每季度的列重排顺序也存一下（你写论文会用到）要这个dtmlda
    pd.DataFrame(lda_orders_each_q, columns=["Quarter", "lda_col_order"]).to_csv(
        OUT_DIR / "per_quarter_lda_order.csv", index=False, encoding="utf-8-sig"
    )

    print("🎉 DONE. 输出目录：", OUT_DIR.resolve())
    print("✅ 推荐用于论文的图：heatmap_mean_matched_norm01_teacher_color.png")


if __name__ == "__main__":
    main()
