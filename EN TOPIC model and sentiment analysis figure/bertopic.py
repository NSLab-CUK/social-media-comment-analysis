# -*- coding: utf-8 -*-bertopic
import os, re, gc, time, warnings, logging as log, traceback
from pathlib import Path
import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ========== 稳定性 ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
log.basicConfig(level=log.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
np.random.seed(42)

# ========== 路径与超参 ==========
IN_CSV = Path("prepared_corpus.csv")
OUT_DIR = Path("EN_BERTopic_Out")

K_TARGET = 6
TOPN_WORDS = 30
SEED = 42
OVERWRITE = True

DEVICE = "cpu"
# 请确保路径正确
SBERT = r"C:\Users\Administrator\Desktop\BS-test\agent_test\表现测试\models\bge-large-en-v1.5"
BATCH_ENC = 64

# --- 定制停用词 ---
my_stop_words = set(ENGLISH_STOP_WORDS)
my_stop_words.update(["with", "all", "out", "some", "what", "can", "use", "its", "im"])
keep_words = {"ai", "intelligence", "artificial", "human", "learning", "data"}
my_stop_words = my_stop_words - keep_words

UMAP_KW = dict(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)

# 【核心配置】：min_df=1 保证不崩
VEC_KW = dict(
    min_df=1,
    max_df=1.0,
    token_pattern=r"(?u)\b\w[\w_\-\+]+\b",
    stop_words=list(my_stop_words)
)

_QRE = re.compile(r'^\s*(\d{4})\s*Q\s*([1-4])\s*$', re.I)


def qkey(q: str):
    m = _QRE.match(str(q));
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)


def load_prep() -> pd.DataFrame:
    assert IN_CSV.exists(), f"找不到输入：{IN_CSV}"
    df = pd.read_csv(IN_CSV)
    df = df[df["text_raw"].astype(str).str.strip().ne("")]
    df = df[df["text_norm"].astype(str).str.strip().ne("")]
    df["quarter"] = df["quarter"].astype(str).replace({"2025Q4": "2025Q3"})
    return df


def extract_topics_df(model: BERTopic, k_expect: int):
    info = model.get_topic_info()
    valid_ids = sorted([int(t) for t in info.Topic.tolist() if int(t) != -1])
    rows = []
    for tid in valid_ids:
        topic = model.get_topic(tid) or []
        for r, (w, wt) in enumerate(topic[:TOPN_WORDS], 1):
            rows.append({"topic": tid, "rank": r, "word": w, "weight": float(wt)})
    return pd.DataFrame(rows)


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for tid, g in df_long.groupby("topic"):
        g2 = g.sort_values("rank")
        row = {"topic": int(tid), "auto_label": f"Topic {int(tid)}"}
        for i, w in enumerate(g2["word"].tolist()[:TOPN_WORDS], 1):
            row[f"word_{i}"] = w
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("topic").reset_index(drop=True)


def build_model(docs_norm, emb_raw, k_topics):
    vec = CountVectorizer(**VEC_KW)

    log.info(f"使用 KMeans 强制聚类为 {k_topics} 个主题...")
    cluster_model = KMeans(n_clusters=k_topics, random_state=SEED, n_init=10)
    umap_model = UMAP(**UMAP_KW)

    model = BERTopic(
        embedding_model=None,
        vectorizer_model=vec,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=True,
        nr_topics=None,
        low_memory=True
    )

    topics, probs = model.fit_transform(docs_norm, embeddings=emb_raw)
    return model, topics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_prep()

    log.info("[boot] DEVICE=%s ; SBERT=%s", DEVICE, SBERT)
    encoder = SentenceTransformer(SBERT, device=DEVICE)

    df = df.sort_values("quarter", key=lambda x: x.map(qkey))
    docs_raw = df["text_raw"].astype(str).tolist()
    docs_norm = df["text_norm"].astype(str).tolist()

    # 原始季度字符串列表
    timestamps_str = df["quarter"].tolist()

    log.info(f"全量文档数: {len(docs_norm)}")
    log.info("Computing Global Embeddings...")
    emb = encoder.encode(
        docs_raw, batch_size=BATCH_ENC, show_progress_bar=True,
        device=DEVICE, convert_to_numpy=True, normalize_embeddings=True
    )

    log.info("Training Global BERTopic Model (min_df=1)...")
    model, topics = build_model(docs_norm, emb, K_TARGET)

    df_global = extract_topics_df(model, K_TARGET)
    df_global.to_csv(OUT_DIR / "bert_topics_global.csv", index=False, encoding="utf-8-sig")

    log.info("Calculating Topics Over Time (Dynamic)...")

    # 建立映射: "2023Q1" -> 0, "2023Q2" -> 1
    unique_quarters = sorted(list(set(timestamps_str)), key=qkey)
    q_to_int = {q: i for i, q in enumerate(unique_quarters)}
    timestamps_int = [q_to_int[t] for t in timestamps_str]

    # 【关键修改】：nr_bins=None
    # 这告诉 BERTopic: "不要去计算区间，直接按照我给你的整数(0,1,2...)进行精确分组"
    topics_over_time = model.topics_over_time(docs_norm, timestamps_int, nr_bins=None)

    # 导出循环
    for step_id, q_label in enumerate(unique_quarters):
        log.info(f"Exporting dynamic topics for {q_label} (Step {step_id})...")

        # 精确匹配 Step ID
        subset = topics_over_time[topics_over_time["Timestamp"] == step_id]

        rows = []
        for _, row in subset.iterrows():
            tid = int(row["Topic"])
            words = row["Words"]
            words_list = words.split(',') if isinstance(words, str) else words

            for r, w in enumerate(words_list[:TOPN_WORDS], 1):
                rows.append({
                    "topic": tid,
                    "rank": r,
                    "word": w.strip(),
                    "weight": 1.0 / (r + 0.5)
                })

        out_long = OUT_DIR / f"bert_topics_{q_label}.csv"
        out_wide = OUT_DIR / f"bert_topics_{q_label}_wide.csv"

        if rows:
            df_q = pd.DataFrame(rows).sort_values(["topic", "rank"])
            df_q.to_csv(out_long, index=False, encoding="utf-8-sig")
            to_wide(df_q).to_csv(out_wide, index=False, encoding="utf-8-sig")
        else:
            log.warning(f"季度 {q_label} 无有效数据 (Step {step_id})。")

    log.info("[done] 输出目录：%s", OUT_DIR.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("[FATAL]\n%s", traceback.format_exc())