# -*- coding: utf-8 -*-
"""
按季度训练 LDA（固定 K=6）
"""


import re, ast, logging as log
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# ========= 路径 & 参数 =========
RAW_INPUT_CSV = Path("weibo_ai_2023_2025_final_clean.csv")
TOKENS_CSV    = Path("weibo_ai_tokens_2023_2025.csv")
OUT_DIR       = Path("LDA_Out")

K             = 6
TOPN_WORDS    = 15
LABEL_TOPN    = 6
SEED          = 42
MIN_DOCS_PER_Q= 20
PASSES        = 10
ITERATIONS    = 200
NO_ABOVE      = 0.5  # 最高 50% 文档出现的高频词剔除

log.basicConfig(level=log.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
np.random.seed(SEED)

# ========= 清洗 & 时间 =========
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
    return pd.to_datetime(s.astype(str).map(_norm), errors="coerce", infer_datetime_format=True)

def quarter_label(dt: pd.Timestamp) -> str:
    return f"{dt.year}Q{(dt.month-1)//3+1}"

def tokenize_fallback(text: str):
    text = re.sub(r'https?://\S+|www\.\S+',' ', str(text))
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5 ]+"," ", text)
    ws = [w for w in text.split() if len(w)>=2 and not w.isdigit()]
    return ws if len(ws)>=2 else []

# ========= 对齐 =========
def load_quarter_docs(raw_csv: Path, tokens_csv: Path):
    df_tok = pd.read_csv(tokens_csv, encoding="utf-8-sig")
    assert "tokens" in df_tok.columns
    toks = []
    for x in df_tok["tokens"].astype(str):
        try:
            arr = ast.literal_eval(x); arr = [str(w).strip() for w in arr if str(w).strip()]
        except: arr=[]
        toks.append(arr)
    df_tok["tokens"] = toks

    df_raw = pd.read_csv(raw_csv, encoding="utf-8-sig")
    text_col = pick_col(df_raw, TEXT_COLS)
    time_col = pick_col(df_raw, TIME_COLS)
    dt = parse_time_series(df_raw[time_col])
    df_raw_ok = df_raw.loc[dt.notna()].copy()
    df_raw_ok["_dt"] = dt.loc[dt.notna()]

    key_tok = (df_tok["clean_text"] if "clean_text" in df_tok.columns
               else df_tok["tokens"].apply(lambda a:" ".join(a)).map(clean_text_func))
    key_raw = df_raw_ok[text_col].astype(str).map(clean_text_func)

    df_map = (pd.DataFrame({"_key": key_raw, "_dt": df_raw_ok["_dt"]})
              .dropna(subset=["_key","_dt"])
              .drop_duplicates(subset=["_key"], keep="first"))

    df_tok2 = df_tok.copy(); df_tok2["_key"] = key_tok
    df_merged = df_tok2.merge(df_map, on="_key", how="left")
    df_merged = df_merged[df_merged["_dt"].notna()].copy()
    df_merged["_q"] = df_merged["_dt"].map(quarter_label)

    by_q_tokens = defaultdict(list)
    for _, r in df_merged.iterrows():
        t = r["tokens"]
        if not isinstance(t, list) or len(t)<2:
            t = tokenize_fallback(str(r["_key"]))
        if t:
            by_q_tokens[r["_q"]].append(t)

    quarters = sorted(by_q_tokens.keys(), key=lambda x: (int(x[:4]), int(x[-1])))
    total_docs = sum(len(v) for v in by_q_tokens.values())
    log.info("[align] LDA 对齐成功文档=%d | 季度=%d", total_docs, len(quarters))
    return quarters, by_q_tokens

# ========= 保存长表 + 宽表 =========
def save_topics(df_long: pd.DataFrame, out_csv_long: Path):
    df_long = df_long.sort_values(["topic","rank"]).reset_index(drop=True)
    out_csv_long.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_csv_long, index=False, encoding="utf-8-sig")

    rows = []
    for tid, g in df_long.groupby("topic"):
        g2 = g.sort_values("rank").head(TOPN_WORDS)
        words = g2["word"].tolist()
        weights = g2["weight"].tolist()
        row = {"topic": int(tid), "label": "、".join(words[:LABEL_TOPN])}
        for i, (w, wt) in enumerate(zip(words,weights), 1):
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
    quarters, by_q_tokens = load_quarter_docs(RAW_INPUT_CSV, TOKENS_CSV)

    for q in quarters:
        docs = by_q_tokens[q]
        n = len(docs)
        log.info("== %s | 文档数=%d ==", q, n)
        if n < MIN_DOCS_PER_Q:
            log.warning("%s 文档少（%d），主题可能不稳定。", q, n)

        # 词典/语料
        dictionary = corpora.Dictionary(docs)
        no_below = max(5, int(n * 0.005))    # ≥0.5% 文档出现
        dictionary.filter_extremes(no_below=no_below, no_above=NO_ABOVE)
        corpus = [dictionary.doc2bow(t) for t in docs]
        if len(dictionary) == 0 or len(corpus) == 0:
            log.warning("%s 词典或语料为空，跳过。", q); continue

        lda = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=K,
            random_state=SEED, passes=PASSES, iterations=ITERATIONS,
            chunksize=4000, alpha='asymmetric', eta='auto',
            minimum_probability=0.0
        )

        rows=[]
        for tid in range(K):
            for r,(w,wt) in enumerate(lda.show_topic(tid, topn=TOPN_WORDS),1):
                rows.append({"topic":tid,"rank":r,"word":w,"weight":float(wt)})

        out_long = OUT_DIR / f"lda_topics_{q}.csv"
        save_topics(pd.DataFrame(rows), out_long)

    log.info("[done] LDA 导出完成 → %s", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
