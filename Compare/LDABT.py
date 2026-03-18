# compare_cn_lda_vs__bertopic_teacher_style.py
# -*- coding: utf-8 -*- lda bt

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


# ===================== 1) Config =====================
BASE_DIR = Path(".")

LDA_TOPIC_DIR = BASE_DIR / "LDA_Out"                # lda_topics_2023Q1.csv ...
BERT_GLOBAL_CSV = BASE_DIR / "bert_topics.csv"

OUT_DIR = BASE_DIR / "Compare_CN_LDA_BERT_Out"

EMBED_MODEL_NAME = "BAAI/bge-m3"
TOPN_FOR_TEXT = 30

CMAP = "YlGnBu"
THEME = "white"
FIGSIZE = (10, 10)
DPI = 220
ANNOT_WHITE_TH = 0.65

RENORM_AFTER_CENTER = True

DOMAIN_STOPWORDS = {
    "人工智能", "ai", "a.i", "智能", "技术", "科技", "发展", "未来", "研究", "算法", "模型", "数据",
    "系统", "应用", "能力", "训练", "学习", "网络", "平台", "产业", "行业", "公司", "企业",
    "中国", "美国", "世界", "人类", "社会", "问题", "可能", "现在", "如果", "就是", "这个", "那个",
    "gpt", "chatgpt", "openai", "大模型", "语言模型"
}

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 2) Utils =====================
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
        raise ValueError(f"{path.name} missing topic column.")

    word_cols = [c for c in df.columns if re.fullmatch(r"word_\d+", str(c))]
    if not word_cols:
        raise ValueError(f"{path.name} looks wide but no word_1.. columns.")

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


def list_lda_quarters(lda_dir: Path):
    qs = []
    for p in lda_dir.glob("lda_topics_*.csv"):
        m = re.match(r"lda_topics_(\d{4}Q[1-4])\.csv$", p.name)
        if m:
            qs.append(m.group(1))
    return sorted(qs, key=lambda x: (int(x[:4]), int(x[-1])))


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def plot_heatmap_like_teacher(mat, title, out_png, row_labels, col_labels, xname="BERTopic Topics"):
    assert sns is not None, "seaborn required for teacher-style heatmap."

    sns.set_theme(style=THEME)
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    ax = sns.heatmap(
        mat,
        annot=True, fmt=".2f",
        annot_kws={"size": 20},
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
    ax.set_ylabel("LDA Topics", fontsize=20)
    plt.xticks(rotation=35, ha="right", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print("saved:", out_png.resolve())


# ===================== 3) Main =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert LDA_TOPIC_DIR.exists(), f"Missing LDA dir: {LDA_TOPIC_DIR.resolve()}"
    assert BERT_GLOBAL_CSV.exists(), f"Missing BERTopic file: {BERT_GLOBAL_CSV.resolve()}"

    quarters = list_lda_quarters(LDA_TOPIC_DIR)
    if not quarters:
        raise RuntimeError("No lda_topics_<Q>.csv found.")
    print("quarters:", quarters)

    bert_df = load_topics_any_format(BERT_GLOBAL_CSV)
    bert_ids = sorted(bert_df["topic"].astype(int).unique().tolist())
    bert_texts = [build_topic_text_from_df(bert_df, t, topn=TOPN_FOR_TEXT) for t in bert_ids]

    device = detect_device()
    print(f"Loading SentenceTransformer: {EMBED_MODEL_NAME} | device={device}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    bert_emb = model.encode(bert_texts, normalize_embeddings=False)

    quarter_records = []
    all_lda_embs = []

    for q in quarters:
        p = LDA_TOPIC_DIR / f"lda_topics_{q}.csv"
        lda_df = load_topics_any_format(p)
        lda_ids = sorted(lda_df["topic"].astype(int).unique().tolist())
        lda_texts = [build_topic_text_from_df(lda_df, t, topn=TOPN_FOR_TEXT) for t in lda_ids]
        lda_emb = model.encode(lda_texts, normalize_embeddings=False)

        quarter_records.append((q, lda_ids, lda_emb))
        all_lda_embs.append(lda_emb)

    global_mean_vec = np.mean(np.vstack([bert_emb] + all_lda_embs), axis=0)

    sims = []
    lda_ids_ref = None
    for q, lda_ids, lda_emb in quarter_records:
        if lda_ids_ref is None:
            lda_ids_ref = lda_ids
        lda_c = lda_emb - global_mean_vec
        bert_c = bert_emb - global_mean_vec
        if RENORM_AFTER_CENTER:
            lda_c = l2_normalize_rows(lda_c)
            bert_c = l2_normalize_rows(bert_c)
        sims.append(cosine_similarity(lda_c, bert_c))

    mean_sim = np.stack(sims, axis=0).mean(axis=0)
    mn, mx = float(mean_sim.min()), float(mean_sim.max())
    mean_sim_norm = (mean_sim - mn) / (mx - mn + 1e-12)

    pd.DataFrame(mean_sim).to_csv(OUT_DIR / "mean_sim_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_sim_norm).to_csv(OUT_DIR / "mean_sim_norm01.csv", index=False, encoding="utf-8-sig")

    row_labels = [f"LDA {t}" for t in lda_ids_ref]
    col_labels = [f"BERTopic {t}" for t in bert_ids]

    plot_heatmap_like_teacher(
        mean_sim_norm,
        title="Topic Correspondence: Quarterly LDA vs Global BERTopic\n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels,
        xname="BERTopic Topics"
    )

    # Optional: matched visualization
    if linear_sum_assignment is not None:
        r_ind, c_ind = linear_sum_assignment(-mean_sim_norm)
        matched = mean_sim_norm[r_ind][:, c_ind]
        row_labels_m = [row_labels[i] for i in r_ind]
        col_labels_m = [col_labels[j] for j in c_ind]

        plot_heatmap_like_teacher(
            matched,
            title="Topic Correspondence: Quarterly LDA vs Quarterly BERTopic \n(0=Least Similar, 1=Most Similar)",
            out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color_matched.png",
            row_labels=row_labels_m,
            col_labels=col_labels_m,
            xname="BERTopic Topics"
        )

    print("DONE. Output:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
