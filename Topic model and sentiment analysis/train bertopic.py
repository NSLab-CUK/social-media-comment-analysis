# train_bertopic_global_fixed6.py
# -*- coding: utf-8 -*-

import re, ast, logging as log
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch



# ========= 路径 & 参数 =========
RAW_INPUT_CSV = Path("weibo_ai_2023_2025_final_clean.csv")
TOKENS_CSV    = Path("weibo_ai_tokens_2023_2025.csv")
OUT_DIR       = Path("LDA_BERTopic_Out")

K_TARGET        = 6
TOPN_WORDS      = 15
LABEL_TOPN      = 6
MIN_DOCS_GLOBAL = 200   # 全局最好别太少；只是提醒，不强制
SEED            = 42
OVERWRITE       = True

# ========= 日志 =========
log.basicConfig(level=log.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

# ========= 清洗：与前述脚本保持一致 =========
_re_url   = re.compile(r'https?://\S+|www\.\S+')
_re_at    = re.compile(r'@[\w\-\u4e00-\u9fff]+')
_re_topic = re.compile(r'#([^#]+)#')
_re_space = re.compile(r'\s+')
_re_keep  = re.compile(r'[A-Za-z0-9\u4e00-\u9fa5]+')

def clean_text_func(s: str) -> str:
    if not isinstance(s, str): return ""
    s = _re_url.sub(" ", s); s = _re_at.sub(" ", s); s = _re_topic.sub(" ", s)
    s = _re_space.sub(" ", s).strip()
    parts = _re_keep.findall(s)
    return " ".join(parts)

# ========= 设备 =========
np.random.seed(SEED)
try:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except Exception:
    pass

DEVICE = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
         else ("cuda" if torch.cuda.is_available() else "cpu")

# 和季度脚本保持一致（最稳）
EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)

# ========= 列 & 时间 =========
TEXT_COLS = ["内容","文本","text","微博正文","content","full_text"]
TIME_COLS = ["时间","发布时间","微博时间","created_at","publish_time","time","date","日期","发布时间(北京时间)"]

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    raise ValueError(f"未找到列，候选：{cands}；现有：{list(df.columns)}")

def parse_time_series(s: pd.Series) -> pd.Series:
    def _norm(x: str) -> str:
        if not isinstance(x, str): x = str(x)
        x = x.strip().replace("年","-").replace("月","-").replace("日"," ")
        x = x.replace("/", "-").replace(".", "-")
        return re.sub(r"\s+"," ", x)
    return pd.to_datetime(s.astype(str).map(_norm), errors="coerce", )

def tokenize_fallback(text: str):
    text = re.sub(r'https?://\S+|www\.\S+',' ', str(text))
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5 ]+"," ", text)
    ws = [w for w in text.split() if len(w)>=2 and not w.isdigit()]
    return ws if len(ws)>=2 else []

# ========= 对齐（raw→去重映射，避免放大） =========
def load_all_docs(raw_csv: Path, tokens_csv: Path):
    assert raw_csv.exists() and tokens_csv.exists()

    df_tok = pd.read_csv(tokens_csv, encoding="utf-8-sig")
    assert "tokens" in df_tok.columns, "tokens CSV 必须含 tokens 列"

    # 解析 tokens
    toks = []
    for x in df_tok["tokens"].astype(str):
        try:
            arr = ast.literal_eval(x)
            arr = [str(w).strip() for w in arr if str(w).strip()]
        except:
            arr = []
        toks.append(arr)
    df_tok["tokens"] = toks
    df_tok["tok_len"] = df_tok["tokens"].apply(len)

    # raw + 时间
    df_raw = pd.read_csv(raw_csv, encoding="utf-8-sig")
    text_col = pick_col(df_raw, TEXT_COLS)
    time_col = pick_col(df_raw, TIME_COLS)
    dt = parse_time_series(df_raw[time_col])
    df_raw_ok = df_raw.loc[dt.notna()].copy()
    df_raw_ok["_dt"] = dt.loc[dt.notna()]

    # 两边统一 clean_text 生成 key
    key_tok = (df_tok["clean_text"] if "clean_text" in df_tok.columns
               else df_tok["tokens"].apply(lambda a:" ".join(a)).map(clean_text_func))
    key_raw = df_raw_ok[text_col].astype(str).map(clean_text_func)

    # raw 映射：每个 _key 只保留第一条时间，避免一对多放大
    df_map = (pd.DataFrame({"_key": key_raw, "_dt": df_raw_ok["_dt"]})
              .dropna(subset=["_key","_dt"])
              .drop_duplicates(subset=["_key"], keep="first"))

    df_tok2 = df_tok.copy()
    df_tok2["_key"] = key_tok

    df_merged = df_tok2.merge(df_map, on="_key", how="left")
    df_merged = df_merged[df_merged["_dt"].notna()].copy()

    docs_str = []
    for _, r in df_merged.iterrows():
        t = r["tokens"]
        if not isinstance(t, list) or len(t) < 2:
            t = tokenize_fallback(str(r["_key"]))
        if t:
            docs_str.append(" ".join(t))

    log.info("[align-global] tokens=%d | raw_ok=%d | raw_unique_key=%d | merged_docs=%d",
             len(df_tok), len(df_raw_ok), len(df_map), len(docs_str))
    return docs_str

# ========= 训练（HDBSCAN→reduce；不足再 KMeans 兜底） =========
def run_bertopic_fixed6_global(docs_str, k_topics=K_TARGET):
    n_docs = len(docs_str)
    if n_docs <= 0:
        return None, pd.DataFrame(columns=["topic","rank","word","weight"])

    if n_docs < k_topics:
        log.warning("样本数(%d) 小于目标主题数(%d)，将用 %d 个主题。", n_docs, k_topics, n_docs)
        k_topics = max(1, n_docs)

    def _extract(model, k_expect=k_topics):
        info = model.get_topic_info()
        valid_ids = [int(t) for t in info.Topic.tolist() if int(t) != -1][:k_expect]
        rows = []
        for new_tid, tid in enumerate(valid_ids):
            topic = model.get_topic(tid) or []
            for r, (w, wt) in enumerate(topic[:TOPN_WORDS], 1):
                rows.append({"topic": new_tid, "rank": r, "word": w, "weight": float(wt)})
        return pd.DataFrame(rows)

    # A) HDBSCAN + reduce
    try:
        vectorizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern=r"(?u)\b\w+\b")
        umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, random_state=SEED)
        hdbscan_model = HDBSCAN(min_cluster_size=max(2, int(n_docs * 0.01)), min_samples=1)

        modelA = BERTopic(
            embedding_model=EMBEDDER,
            vectorizer_model=vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=3,
            calculate_probabilities=False,
            verbose=False,
            nr_topics=None
        )
        topics, _ = modelA.fit_transform(docs_str)

        # 尝试减少离群点、减少主题到 k_topics
        try:
            topics = modelA.reduce_outliers(docs_str, topics)
        except Exception:
            pass
        try:
            modelA.reduce_topics(docs_str, nr_topics=k_topics)
        except Exception:
            pass

        dfA = _extract(modelA, k_expect=k_topics)
        if dfA["topic"].nunique() >= k_topics:
            return modelA, dfA

        log.info("[info] HDBSCAN+reduce 后主题不足，转入 KMeans 兜底。")
    except Exception as e:
        log.warning("BERTopic-HDBSCAN 阶段失败：%s", e)

    # B) KMeans 兜底（兼容旧版 BERTopic：不用 cluster_model）
    try:
        emb = EMBEDDER.encode(
            docs_str, batch_size=64, show_progress_bar=True,
            device=DEVICE, convert_to_numpy=True
        )
        km = KMeans(n_clusters=k_topics, random_state=SEED, n_init=10)
        labels = km.fit_predict(emb)

        vectorizerK = CountVectorizer(min_df=1, max_df=1.0, token_pattern=r"(?u)\b\w+\b")
        modelB = BERTopic(
            embedding_model=EMBEDDER,
            vectorizer_model=vectorizerK,
            calculate_probabilities=False,
            verbose=False,
            nr_topics=None
        )
        topics, _ = modelB.fit_transform(docs_str, embeddings=emb, y=labels)
        dfB = _extract(modelB, k_expect=k_topics)
        return modelB, dfB
    except Exception as e:
        log.warning("BERTopic-KMeans 兜底仍失败：%s", e)
        return None, pd.DataFrame(columns=["topic","rank","word","weight"])

# ========= 保存：长表 + 宽表 =========
def save_topics_both(df_long: pd.DataFrame, out_csv_long: Path):
    df_long = df_long.sort_values(["topic","rank"]).reset_index(drop=True)
    out_csv_long.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_csv_long, index=False, encoding="utf-8-sig")

    rows = []
    for tid, g in df_long.groupby("topic"):
        g2 = g.sort_values("rank").head(TOPN_WORDS)
        words = g2["word"].tolist()
        weights = g2["weight"].tolist()
        row = {"topic": int(tid), "label": "、".join(words[:LABEL_TOPN])}
        for i, (w, wt) in enumerate(zip(words, weights), 1):
            row[f"word_{i}"] = w
            row[f"weight_{i}"] = float(wt)
        rows.append(row)

    df_wide = pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)
    out_csv_wide = out_csv_long.with_name(out_csv_long.stem + "_wide.csv")
    df_wide.to_csv(out_csv_wide, index=False, encoding="utf-8-sig")
    log.info("[save] %s & %s", out_csv_long.name, out_csv_wide.name)

# ========= 主流程 =========
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs_str = load_all_docs(RAW_INPUT_CSV, TOKENS_CSV)
    n = len(docs_str)
    log.info("== GLOBAL | 文档数：%d ==", n)
    if n < MIN_DOCS_GLOBAL:
        log.warning("GLOBAL 样本偏少（%d < %d），主题可能更不稳定。", n, MIN_DOCS_GLOBAL)

    out_long = OUT_DIR / "bert_topics.csv"
    if (not OVERWRITE) and out_long.exists() and out_long.stat().st_size > 0:
        log.info("[skip] 已存在：%s", out_long.name)
        return

    model, df_topics = run_bertopic_fixed6_global(docs_str, k_topics=K_TARGET if n >= 1 else 0)
    if df_topics.empty:
        log.error("[fail] GLOBAL 训练失败/无主题。")
        return

    # 规范 topic 连续编号
    uniq = sorted(df_topics["topic"].unique().tolist())
    remap = {old: i for i, old in enumerate(uniq)}
    df_topics["topic"] = df_topics["topic"].map(remap)

    save_topics_both(df_topics, out_long)
    log.info("[done]  BERTopic 已输出到：%s", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
