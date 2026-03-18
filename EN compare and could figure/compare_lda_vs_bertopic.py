# compare_en_lda_vs_bertopic.py
# -*- coding: utf-8 -*-
"""
EN: Topic Correspondence -  LDA vs  BERTopic
- SentenceTransformer embeddings
- domain stopwords filtering
- global mean centering (remove shared background semantics)
- renorm after center for sharper comparison
- min-max normalize to 0-1 for visualization
- Hungarian matching to reorder for clearer diagonal

Inputs:
  - LDA: TOPIC_OUT/lda_topics_topwords.csv (long: model, topic, rank, word, weight)
  - BERTopic: EN_BERTopic_Out/bert_topics_global.csv (long: topic, rank, word, weight)
Outputs:
  - Compare_EN_LDA_BERT_Out/
      mean_sim_raw.csv
      mean_sim_norm01.csv
      heatmap_mean_norm01_teacher_color.png
      mean_sim_matched_raw.csv
      mean_sim_matched_norm01.csv
      heatmap_mean_matched_norm01_teacher_color.png
      matched_col_order.csv
"""

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


# ===================== 1) CONFIG =====================
BASE_DIR = Path(".")

LDA_GLOBAL = BASE_DIR / "TOPIC_OUT" / "lda_topics_topwords.csv"
BERT_GLOBAL = BASE_DIR / "EN_BERTopic_Out" / "bert_topics_en.csv"

OUT_DIR = BASE_DIR / "Compare_EN_LDA_BERT_Out"

# Embedding model
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

TOPN_FOR_TEXT = 30

# Plot settings
CMAP = "YlGnBu"
THEME = "white"
FIGSIZE = (10, 10)
DPI = 220
ANNOT_WHITE_TH = 0.65

# Make it sharper
RENORM_AFTER_CENTER = True

# Domain stopwords
DOMAIN_STOPWORDS = {
    "ai","artificial","intelligence","learning","data","model","models","technology",
    "human","use","new","time","based","systems","development","potential","future",
    "world","machine"
}

SAVE_RAW = True


# ===================== 2) UTILS =====================
def detect_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _safe_word(w: str):
    w = str(w).strip()
    if (not w) or (w.lower() == "nan"):
        return None
    if w.lower() in DOMAIN_STOPWORDS:
        return None
    return w


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def build_topic_text(df: pd.DataFrame, topic_id: int, topn: int = TOPN_FOR_TEXT) -> str:
    sub = df[df["topic"].astype(int) == int(topic_id)].sort_values("rank").head(topn)
    words = [str(x) for x in sub["word"].tolist()]
    filtered = [x for x in (_safe_word(w) for w in words) if x is not None]
    if not filtered:
        filtered = [w for w in words if str(w).strip() and str(w).lower() != "nan"]
    return " ".join(filtered)


def load_lda_global() -> pd.DataFrame:
    p = Path(LDA_GLOBAL)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find global LDA file: {p.resolve()}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    need = {"topic","rank","word"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p.name} missing columns {need}")
    df = df.copy()
    df["topic"] = df["topic"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["topic","rank"]).reset_index(drop=True)


def load_bert_global() -> pd.DataFrame:
    p = Path(BERT_GLOBAL)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find global BERTopic file: {p.resolve()}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    need = {"topic","rank","word"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p.name} missing columns {need}")
    df = df.copy()
    df["topic"] = df["topic"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["topic","rank"]).reset_index(drop=True)


def plot_heatmap_teacher(mat, title, out_png, row_labels, col_labels, xname="BERTopic Topics"):
    assert sns is not None, "Need seaborn for teacher-style heatmap. pip install seaborn"
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
    print("✅ saved:", out_png.resolve())


# ===================== 3) MAIN =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load global files
    lda_df = load_lda_global()
    bert_df = load_bert_global()

    lda_ids = sorted(lda_df["topic"].unique().tolist())
    bert_ids = sorted(bert_df["topic"].unique().tolist())

    print("✅ Global LDA topics:", lda_ids)
    print("✅ Global BERTopic topics:", bert_ids)

    if len(lda_ids) < 2 or len(bert_ids) < 2:
        raise RuntimeError("❌ Not enough topics in LDA/BERTopic.")

    device = detect_device()
    print(f"Loading SentenceTransformer: {EMBED_MODEL} | device={device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # Encode LDA
    lda_texts = [build_topic_text(lda_df, t) for t in lda_ids]
    lda_emb = model.encode(lda_texts, normalize_embeddings=False)

    # Encode BERTopic
    bert_texts = [build_topic_text(bert_df, t) for t in bert_ids]
    bert_emb = model.encode(bert_texts, normalize_embeddings=False)

    # Global mean centering
    all_embs = np.vstack([lda_emb, bert_emb])
    global_mean_vec = np.mean(all_embs, axis=0)

    lda_c = lda_emb - global_mean_vec
    bert_c = bert_emb - global_mean_vec

    if RENORM_AFTER_CENTER:
        lda_c = l2_normalize_rows(lda_c)
        bert_c = l2_normalize_rows(bert_c)

    # Compute similarity (LDA x BERTopic)
    sim = cosine_similarity(lda_c, bert_c)

    # Normalize to 0-1
    mn, mx = float(sim.min()), float(sim.max())
    sim_norm = (sim - mn) / (mx - mn + 1e-12)

    if SAVE_RAW:
        pd.DataFrame(sim).to_csv(OUT_DIR / "mean_sim_raw.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sim_norm).to_csv(OUT_DIR / "mean_sim_norm01.csv", index=False, encoding="utf-8-sig")

    row_labels = [f"LDA {t}" for t in lda_ids]
    col_labels = [f"BERT {t}" for t in bert_ids]

    plot_heatmap_teacher(
        sim_norm,
        title="Topic Correspondence: Quarterly LDA vs Quarterly BERTopic\n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels,
        xname="BERTopic Topics"
    )

    # ===== Matched (Hungarian) =====
    assert linear_sum_assignment is not None, "Need scipy for matching: pip install scipy"

    r_ind, c_ind = linear_sum_assignment(-sim)  # maximize similarity
    sim_m = sim[r_ind][:, c_ind]
    mn2, mx2 = float(sim_m.min()), float(sim_m.max())
    sim_m_norm = (sim_m - mn2) / (mx2 - mn2 + 1e-12)

    if SAVE_RAW:
        pd.DataFrame(sim_m).to_csv(OUT_DIR / "mean_sim_matched_raw.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sim_m_norm).to_csv(OUT_DIR / "mean_sim_matched_norm01.csv", index=False, encoding="utf-8-sig")

    col_labels_m = [f"BERT {bert_ids[j]}" for j in c_ind]
    row_labels_m = [f"LDA {lda_ids[i]}" for i in r_ind]

    pd.DataFrame({"lda_row": r_ind, "bert_col": c_ind}).to_csv(
        OUT_DIR / "matched_col_order.csv", index=False, encoding="utf-8-sig"
    )

    plot_heatmap_teacher(
        sim_m_norm,
        title="Topic Correspondence: Quarterly LDA vs Quarterly BERTopic \n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_matched_norm01_teacher_color.png",
        row_labels=row_labels_m,
        col_labels=col_labels_m,
        xname="BERTopic Topics"
    )

    print("🎉 DONE ->", OUT_DIR.resolve())
    print("✅ Recommend for paper:", (OUT_DIR / "heatmap_mean_matched_norm01_teacher_color.png").name)


if __name__ == "__main__":
    main()
