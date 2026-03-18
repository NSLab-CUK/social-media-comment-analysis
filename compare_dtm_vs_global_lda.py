# compare_en_dtm_vs_global_lda_teacher_style_matched.py
# -*- coding: utf-8 -*-
"""
EN: Topic Correspondence - Dynamic DTM (per quarter) vs Global LDA (single file)
- SentenceTransformer embeddings
- domain stopwords filtering (optional)
- global mean centering (remove shared background semantics)
- (optional) renorm after center -> sharper
- per-quarter sim, then mean over quarters
- min-max normalize to 0-1 for visualization
- Hungarian matching to reorder LDA columns for clearer diagonal

Inputs:
  - DTM: TOPIC_OUT/dtm_topics_2023Q1.csv ... dtm_topics_2025Q3.csv  (long: topic, rank, word, weight)
  - LDA: TOPIC_OUT/lda_topics_topwords.csv                         (long: model, topic, rank, word, weight)
Outputs:
  - Compare_EN_LDA_DTM_Out/
      mean_sim_raw.csv
      mean_sim_norm01.csv
      heatmap_mean_norm01_teacher_color.png
      mean_sim_matched_raw.csv
      mean_sim_matched_norm01.csv
      heatmap_mean_matched_norm01_teacher_color.png
      matched_col_order.csv
"""

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


# ===================== 1) CONFIG (edit here only) =====================
BASE_DIR = Path(".")

DTM_DIR = BASE_DIR / "TOPIC_OUT"                 # dtm_topics_2023Q1.csv ...
LDA_GLOBAL = DTM_DIR / "lda_topics_topwords.csv" # single global lda file

OUT_DIR = BASE_DIR / "Compare_EN_LDA_DTM_Out"

# You can use local model path or HF name:
# EMBED_MODEL = r"C:\...\models\bge-large-en-v1.5"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

QUARTERS = [
    "2023Q1","2023Q2","2023Q3","2023Q4",
    "2024Q1","2024Q2","2024Q3","2024Q4",
    "2025Q1","2025Q2","2025Q3",
]

TOPN_FOR_TEXT = 30

# teacher-like plot
CMAP = "YlGnBu"
THEME = "white"
FIGSIZE = (10, 10)
DPI = 220
ANNOT_WHITE_TH = 0.65

# make it sharper
RENORM_AFTER_CENTER = True

# domain stopwords (optional, adjust if needed)
DOMAIN_STOPWORDS = {
    "ai","artificial","intelligence","learning","data","model","models","technology",
    "human","use","new","time","based","systems","development","potential","future",
    "world","machine"
}

SAVE_RAW = True


# ===================== 2) utils =====================
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


def load_dtm_quarter(q: str) -> pd.DataFrame:
    p = DTM_DIR / f"dtm_topics_{q}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["topic","rank","word","weight"])
    df = pd.read_csv(p, encoding="utf-8-sig")
    need = {"topic","rank","word"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p.name} missing columns {need}")
    df = df.copy()
    df["topic"] = df["topic"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["topic","rank"]).reset_index(drop=True)


def load_lda_global() -> pd.DataFrame:
    p = Path(LDA_GLOBAL)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find global LDA file: {p.resolve()}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    # your file has: model, topic, rank, word, weight
    need = {"topic","rank","word"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p.name} missing columns {need}")
    df = df.copy()
    df["topic"] = df["topic"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return df.sort_values(["topic","rank"]).reset_index(drop=True)


def plot_heatmap_teacher(mat, title, out_png, row_labels, col_labels, xname="LDA Topics"):
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
    ax.set_ylabel("DTM Topics", fontsize=20)
    plt.xticks(rotation=35, ha="right",fontsize=18)
    plt.yticks(rotation=0,fontsize=18)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print("✅ saved:", out_png.resolve())


# ===================== 3) main =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load global LDA
    lda_df = load_lda_global()
    lda_ids = sorted(lda_df["topic"].unique().tolist())
    print("✅ Global LDA topics:", lda_ids)

    # load DTM quarters available
    quarter_records = []
    for q in QUARTERS:
        dtm_df = load_dtm_quarter(q)
        if dtm_df.empty:
            continue
        dtm_ids = sorted(dtm_df["topic"].unique().tolist())
        quarter_records.append((q, dtm_df, dtm_ids))

    if not quarter_records:
        raise RuntimeError("❌ No valid DTM quarterly files found. Check dtm_topics_<Q>.csv paths.")

    dtm_ids_ref = quarter_records[0][2]
    print("✅ DTM topics (ref):", dtm_ids_ref)
    if len(lda_ids) < 2 or len(dtm_ids_ref) < 2:
        raise RuntimeError("❌ Not enough topics in LDA/DTM.")

    device = detect_device()
    print(f"Loading SentenceTransformer: {EMBED_MODEL} | device={device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # encode LDA once
    lda_texts = [build_topic_text(lda_df, t) for t in lda_ids]
    lda_emb = model.encode(lda_texts, normalize_embeddings=False)

    # encode all DTM quarters (for global centering)
    all_embs = [lda_emb]
    dtm_embs_by_q = []

    for q, dtm_df, dtm_ids in quarter_records:
        dtm_texts = [build_topic_text(dtm_df, t) for t in dtm_ids]
        dtm_emb = model.encode(dtm_texts, normalize_embeddings=False)
        dtm_embs_by_q.append((q, dtm_ids, dtm_emb))
        all_embs.append(dtm_emb)

    global_mean_vec = np.mean(np.vstack(all_embs), axis=0)

    # ===== A) unaligned mean =====
    sims = []
    for q, dtm_ids, dtm_emb in dtm_embs_by_q:
        dtm_c = dtm_emb - global_mean_vec
        lda_c = lda_emb - global_mean_vec
        if RENORM_AFTER_CENTER:
            dtm_c = l2_normalize_rows(dtm_c)
            lda_c = l2_normalize_rows(lda_c)
        sims.append(cosine_similarity(dtm_c, lda_c))

    mean_sim = np.stack(sims, axis=0).mean(axis=0)
    mn, mx = float(mean_sim.min()), float(mean_sim.max())
    mean_sim_norm = (mean_sim - mn) / (mx - mn + 1e-12)

    if SAVE_RAW:
        pd.DataFrame(mean_sim).to_csv(OUT_DIR / "mean_sim_raw.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(mean_sim_norm).to_csv(OUT_DIR / "mean_sim_norm01.csv", index=False, encoding="utf-8-sig")

    row_labels = [f"DTM {t}" for t in dtm_ids_ref]
    col_labels = [f"LDA {t}" for t in lda_ids]

    plot_heatmap_teacher(
        mean_sim_norm,
        title="Topic Correspondence: Dynamic DTM vs Quarterly LDA\n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels,
        xname="LDA Topics"
    )

    # ===== B) matched (Hungarian) =====
    assert linear_sum_assignment is not None, "Need scipy for matching: pip install scipy"

    r_ind, c_ind = linear_sum_assignment(-mean_sim)  # maximize similarity
    mean_sim_m = mean_sim[r_ind][:, c_ind]
    mn2, mx2 = float(mean_sim_m.min()), float(mean_sim_m.max())
    mean_sim_m_norm = (mean_sim_m - mn2) / (mx2 - mn2 + 1e-12)

    if SAVE_RAW:
        pd.DataFrame(mean_sim_m).to_csv(OUT_DIR / "mean_sim_matched_raw.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(mean_sim_m_norm).to_csv(OUT_DIR / "mean_sim_matched_norm01.csv", index=False, encoding="utf-8-sig")

    col_labels_m = [f"LDA {lda_ids[j]}" for j in c_ind]
    row_labels_m = [f"DTM {dtm_ids_ref[i]}" for i in r_ind]

    pd.DataFrame({"dtm_row": r_ind, "lda_col": c_ind}).to_csv(
        OUT_DIR / "matched_col_order.csv", index=False, encoding="utf-8-sig"
    )

    plot_heatmap_teacher(
        mean_sim_m_norm,
        title="Topic Correspondence: Dynamic DTM vs Quarterly LDA \n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_matched_norm01_teacher_color.png",
        row_labels=row_labels_m,
        col_labels=col_labels_m,
        xname="LDA Topics"
    )

    print("🎉 DONE ->", OUT_DIR.resolve())
    print("✅ Recommend for paper:", (OUT_DIR / "heatmap_mean_matched_norm01_teacher_color.png").name)


if __name__ == "__main__":
    main()
