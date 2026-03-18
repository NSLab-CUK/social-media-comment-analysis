# compare_cn_dtm_vs bertopic_teacher_style_DTM_YlGnBu.py
# -*- coding: utf-8 -*-

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

DTM_MODEL_PATH = BASE_DIR / "my_dtm_model.gensim"
TRUE_TIMES_CSV = BASE_DIR / "cn_topic_sentiment_out_new" / "CN_DTM_topic_times.csv"
DTM_TOPIC_DIR  = BASE_DIR / "CN_DTM_TopicWords_Out"

BERT_GLOBAL_CSV = BASE_DIR / "bert_topics.csv"
OUT_DIR = BASE_DIR / "Compare_CN_Out"

EMBED_MODEL_NAME = "BAAI/bge-m3"
TOPN_FOR_TEXT = 30

# ✅ 你要的颜色：淡黄→绿→蓝
CMAP = "YlGnBu"
THEME = "white"

# ✅ 你喜欢的“方正大小”
FIGSIZE = (10, 10)
DPI = 220

# ✅ 高于这个阈值就用白字（调到你觉得最像为止）
ANNOT_WHITE_TH = 0.65

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


def load_quarters_from_true(true_csv: Path):
    df = pd.read_csv(true_csv, encoding="utf-8-sig")
    if "Quarter" not in df.columns:
        qcol = next((c for c in df.columns if c.lower() == "quarter"), None)
        if not qcol:
            raise ValueError(f"{true_csv.name} 需要 Quarter 列。现有列：{list(df.columns)}")
        df = df.rename(columns={qcol: "Quarter"})
    quarters = df["Quarter"].astype(str).tolist()
    return [q.strip().upper().replace(" ", "") for q in quarters]


def export_dtm_topics_if_needed(dtm_model_path: Path, quarters: list, out_dir: Path, topn_words: int = 15):
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in quarters:
        p = out_dir / f"dtm_topics_{q}.csv"
        if not p.exists() or p.stat().st_size == 0:
            break
    else:
        print("✅ 已存在 DTM 每季度 topic 词表，跳过导出。")
        return

    print("⚙️ 未发现完整 dtm_topics_<Q>.csv，将从 DTM 模型导出…")
    from gensim.models.ldaseqmodel import LdaSeqModel
    dtm = LdaSeqModel.load(str(dtm_model_path))

    K = int(dtm.num_topics)
    T = int(dtm.num_time_slices)
    assert len(quarters) == T, f"TRUE_TIMES 的季度数({len(quarters)})必须等于 time_slices({T})"

    pat = re.compile(r'([0-9]*\.?[0-9]+)\s*\*\s*"?([^"\+]+?)"?\s*(?:\+|$)')

    for t, q in enumerate(quarters):
        rows = []
        for k in range(K):
            s = dtm.print_topic(topic=k, time=t, top_terms=topn_words)
            if isinstance(s, (list, tuple)):
                s = " + ".join([str(x) for x in s])
            s = str(s)

            pairs = pat.findall(s)
            if not pairs:
                pairs = re.findall(r'([0-9]*\.?[0-9]+)\s*\*\s*"?([^"\s\+]+)"?', s)

            tmp = []
            for wt, w in pairs:
                w = str(w).strip()
                if not w:
                    continue
                try:
                    wt = float(wt)
                except Exception:
                    wt = 0.0
                tmp.append((w, wt))
            tmp.sort(key=lambda x: x[1], reverse=True)

            for r, (w, wt) in enumerate(tmp[:topn_words], 1):
                rows.append({"topic": k, "rank": r, "word": w, "weight": float(wt)})

        out_path = out_dir / f"dtm_topics_{q}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print("✅ saved:", out_path)

    print("🎉 DTM topic 词表导出完成。")


def plot_heatmap_like_teacher(mat, title, out_png, row_labels, col_labels):
    assert sns is not None, "需要 seaborn 才能画出你这张图的同款风格。"

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

    # ✅ 自动把高值格子标注改成白字（非常像你那张图）
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except Exception:
            continue
        text.set_color("white" if val >= ANNOT_WHITE_TH else "black")

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("BERTopic Topics", fontsize=20)
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
    DTM_TOPIC_DIR.mkdir(parents=True, exist_ok=True)

    quarters = load_quarters_from_true(TRUE_TIMES_CSV)
    print("✅ quarters (aligned):", quarters)

    export_dtm_topics_if_needed(DTM_MODEL_PATH, quarters, DTM_TOPIC_DIR, topn_words=15)

    assert BERT_GLOBAL_CSV.exists(), f"找不到全局 BERTopic 文件：{BERT_GLOBAL_CSV.resolve()}"
    bert_df = load_topics_any_format(BERT_GLOBAL_CSV)
    bert_ids = sorted(bert_df["topic"].astype(int).unique().tolist())
    bert_texts = [build_topic_text_from_df(bert_df, t, topn=TOPN_FOR_TEXT) for t in bert_ids]
    print(f"✅ loaded global BERTopic: K={len(bert_ids)}")

    device = detect_device()
    print(f"Loading SentenceTransformer: {EMBED_MODEL_NAME} | device={device}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    bert_emb = model.encode(bert_texts, normalize_embeddings=False)

    quarter_records = []
    all_dtm_embs = []

    for q in quarters:
        p = DTM_TOPIC_DIR / f"dtm_topics_{q}.csv"
        if not p.exists() or p.stat().st_size == 0:
            print("⚠️ missing:", p.name)
            continue
        dtm_df = load_topics_any_format(p)
        dtm_ids = sorted(dtm_df["topic"].astype(int).unique().tolist())
        dtm_texts = [build_topic_text_from_df(dtm_df, t, topn=TOPN_FOR_TEXT) for t in dtm_ids]
        dtm_emb = model.encode(dtm_texts, normalize_embeddings=False)

        quarter_records.append((q, dtm_ids, dtm_emb))
        all_dtm_embs.append(dtm_emb)

    if not quarter_records:
        raise RuntimeError("没有任何季度的 DTM topic 文件可用。")

    global_mean_vec = np.mean(np.vstack([bert_emb] + all_dtm_embs), axis=0)
    bert_centered = bert_emb - global_mean_vec

    sims = []
    dtm_ids_ref = None

    for q, dtm_ids, dtm_emb in quarter_records:
        if dtm_ids_ref is None:
            dtm_ids_ref = dtm_ids
        if dtm_ids != dtm_ids_ref:
            print(f"⚠️ {q} DTM topic 编号不一致：{dtm_ids} vs {dtm_ids_ref}（会影响平均）")

        dtm_centered = dtm_emb - global_mean_vec
        sims.append(cosine_similarity(dtm_centered, bert_centered))

    mean_sim = np.stack(sims, axis=0).mean(axis=0)

    # min-max -> 0..1（可视化）
    mn, mx = float(mean_sim.min()), float(mean_sim.max())
    mean_sim_norm = (mean_sim - mn) / (mx - mn + 1e-12)

    pd.DataFrame(mean_sim).to_csv(OUT_DIR / "mean_sim_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_sim_norm).to_csv(OUT_DIR / "mean_sim_norm01.csv", index=False, encoding="utf-8-sig")

    # ✅ 轴标签只显示编号（你要的）
    row_labels = [f"DTM {t}" for t in dtm_ids_ref]
    col_labels = [f"BERTopic {t}" for t in bert_ids]

    plot_heatmap_like_teacher(
        mean_sim_norm,
        title="Topic Correspondence: Dynamic DTM vs Quarterly BERTopic\n(0=Least Similar, 1=Most Similar)",
        out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color.png",
        row_labels=row_labels,
        col_labels=col_labels
    )

    # 匈牙利匹配（展示用）
    if linear_sum_assignment is not None:
        r_ind, c_ind = linear_sum_assignment(-mean_sim_norm)
        matched = mean_sim_norm[r_ind][:, c_ind]
        row_labels_m = [row_labels[i] for i in r_ind]
        col_labels_m = [col_labels[j] for j in c_ind]

        plot_heatmap_like_teacher(
            matched,
            title="Topic Correspondence: Dynamic DTM vs Quarterly BERTopic \n(0=Least Similar, 1=Most Similar)",
            out_png=OUT_DIR / "heatmap_mean_norm01_teacher_color_matched.png",
            row_labels=row_labels_m,
            col_labels=col_labels_m
        )

    print("🎉 DONE. 输出目录：", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

