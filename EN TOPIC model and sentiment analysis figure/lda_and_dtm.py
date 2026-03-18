import os, re, csv, warnings, math, sys, traceback
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore")

# ---------- 路径与参数 ----------
IN_CSV = Path("PREP/prepared_corpus.csv")  # 来自你的预处理脚本
OUT_DIR = Path("TOPIC_OUT")
K = 6  # 固定主题数
SEED = 42

# 【修改】词典过滤参数 (对应 BERTopic: min_df=5, max_df=1.0)
DICT_NO_BELOW = 5  # 对应 min_df=5
DICT_NO_ABOVE = 1.0  # 对应 max_df=1.0 (保留所有高频词)
KEEP_N = 200000

# LDA 训练强度
LDA_PASSES = 40
LDA_ITER = 400
CHUNK_SIZE = 2000

# DTM 训练强度
DTM_BURN_IN = 100
DTM_ITER = 500


# 【新增】构建与 BERTopic 完全一致的停用词表
def get_custom_stopwords():
    s = set(ENGLISH_STOP_WORDS)
    # 1. 去除特定的噪音词
    s.update(["with", "all", "out", "some", "what", "can", "use", "its", "im"])
    # 2. 保护领域核心词 (从停用词中移除，强制保留)
    keep_words = {"ai", "intelligence", "artificial", "human", "learning", "data"}
    return s - keep_words


STOP_WORDS = get_custom_stopwords()

# ---------- 小工具 ----------
_QRE = re.compile(r"^\s*(\d{4})\s*Q\s*([1-4])\s*$", re.I)


def qkey(q: str) -> Tuple[int, int]:
    m = _QRE.match(str(q));
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)


def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_write(msg: str):
    ensure_outdir()
    with open(OUT_DIR / "basic_log.txt", "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg)


def load_prep() -> pd.DataFrame:
    assert IN_CSV.exists(), f"找不到输入文件：{IN_CSV}"
    df = pd.read_csv(IN_CSV)
    # 只使用 text_norm（已合并术语）与 quarter
    need = {"text_norm", "quarter"}
    miss = need - set(df.columns)
    assert not miss, f"缺少必要列：{miss}，请确认来自预处理脚本的输出。"
    df["quarter"] = (df["quarter"].astype(str)
                     .replace({"2025Q4": "2025Q3"}))  # 防御性规范
    # 去除空文本
    df = df[df["text_norm"].astype(str).str.strip().ne("")].copy()
    return df


def texts_to_tokens(texts: List[str]) -> List[List[str]]:
    # 【修改】这里加入停用词过滤，防止 'with' 等词进入模型
    result = []
    for t in texts:
        tokens = str(t).split()
        # 过滤：如果词不在停用词表中，则保留
        tokens = [w for w in tokens if w not in STOP_WORDS]
        result.append(tokens)
    return result


# ---------- LDA（gensim） ----------
def run_lda_k6(tokens_all: List[List[str]]):
    # 词典
    d = corpora.Dictionary(tokens_all)

    # 【新增】应用过滤：min_df=5, max_df=1.0
    d.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE, keep_n=KEEP_N)

    corpus = [d.doc2bow(doc) for doc in tokens_all]
    assert len(d) > 0 and len(corpus) > 0, "词典/语料为空，请检查阈值与数据量。"

    lda = LdaModel(
        corpus=corpus, id2word=d, num_topics=K,
        random_state=SEED, passes=LDA_PASSES, iterations=LDA_ITER,
        chunksize=CHUNK_SIZE, eval_every=None, minimum_probability=0.0,
        alpha="asymmetric", eta="auto"
    )
    lda.save(str(OUT_DIR / "lda_model.model"))
    d.save(str(OUT_DIR / "lda_dictionary.dict"))
    log_write(f"[LDA] 模型已保存至 {OUT_DIR / 'lda_model.model'}")

    # 导出主题 top words
    rows = []
    for k in range(K):
        topic = lda.get_topic_terms(k, topn=20)  # list[(term_id, weight)]
        # 转成(词, 权重)
        words = [(d[id_], float(w)) for id_, w in topic]
        for r, (w, wt) in enumerate(words, 1):
            rows.append({"model": "LDA", "topic": k, "rank": r, "word": w, "weight": wt})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lda_topics_topwords.csv", index=False, encoding="utf-8-sig")
    log_write(f"[LDA] 词典大小={len(d)}，文档数={len(corpus)}；top词已保存：lda_topics_topwords.csv")

    # 简要困惑度（训练集）与 c_v（可选）
    try:
        lp = lda.log_perplexity(corpus)
        perp = math.exp(-lp)
        log_write(f"[LDA] 训练集 log_perplexity={lp:.4f}  perplexity={perp:.2f}（仅供参考）")
    except Exception:
        pass
    try:
        from gensim.models.coherencemodel import CoherenceModel
        cv = CoherenceModel(model=lda, texts=tokens_all, dictionary=d, coherence="c_v",
                            processes=max(1, os.cpu_count() or 2 - 1)).get_coherence()
        log_write(f"[LDA] C_v={cv:.4f}（仅供参考）")
    except Exception:
        pass


# ---------- DTM
def run_dtm_k6_gensim(df_sorted: pd.DataFrame, by_quarter_tokens: Dict[str, List[List[str]]]):
    """
    用 gensim 的 LdaSeqModel 训练 DTM
    """
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from gensim.models.ldaseqmodel import LdaSeqModel

    quarters = sorted(by_quarter_tokens.keys(), key=qkey)
    docs_by_time = [by_quarter_tokens[q] for q in quarters]
    time_slices = [len(d) for d in docs_by_time]

    # 1) 全量词典 & 语料
    all_docs = [doc for docs in docs_by_time for doc in docs]
    d = corpora.Dictionary(all_docs)

    # 【新增】应用过滤：min_df=5, max_df=1.0，与 LDA 保持一致
    d.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE, keep_n=KEEP_N)

    corpus_all = [d.doc2bow(x) for x in all_docs]
    assert len(d) > 0 and len(corpus_all) > 0, "[DTM/gensim] 词典或语料为空，请检查阈值/数据。"

    log_write(f"[DTM/gensim] slices={time_slices} | vocab={len(d)} | docs_total={len(all_docs)}")

    # 2) 先训练一个基础 LDA（用于初始化 DTM 的充分统计量）
    base_lda = LdaModel(
        corpus=corpus_all, id2word=d, num_topics=K,
        random_state=SEED, passes=20, iterations=400,
        alpha="asymmetric", eta="auto", chunksize=2000, eval_every=None
    )


    # 3) 创建 LdaSeqModel
    lds = LdaSeqModel(
        corpus=corpus_all,
        time_slice=time_slices,
        id2word=d,
        num_topics=K,
        initialize="ldamodel",  # 这里使用基础 LDA 的状态初始化
        lda_model=base_lda,  # ← 必须提供
        sstats=None,
        alphas=0.01
    )
    lds.save(str(OUT_DIR / "dtm_model.model"))
    d.save(str(OUT_DIR / "dtm_dictionary.dict"))
    log_write(f"[DTM] 模型已保存至 {OUT_DIR / 'dtm_model.model'}")
    with open(OUT_DIR / "dtm_model.pkl", "wb") as f:
        pickle.dump(lds, f)
    # 4) 导出每个季度的 Top 词
    for t, q in enumerate(quarters):
        rows = []
        for k in range(K):
            words_t = lds.print_topic_times(k, top_terms=20)[t]
            for r, (w, wt) in enumerate(words_t, 1):
                rows.append({
                    "model": "DTM", "quarter": q, "topic": k, "rank": r, "word": w, "weight": float(wt)
                })
        pd.DataFrame(rows).to_csv(OUT_DIR / f"dtm_topics_{q}.csv", index=False, encoding="utf-8-sig")

    log_write("[DTM/gensim] 各季度主题词已保存：dtm_topics_<quarter>.csv")


# ---------- 主流程 ----------
def main():
    try:
        import multiprocessing as mp
        mp.freeze_support()
    except Exception:
        pass

    ensure_outdir()
    df = load_prep()

    # 准备 tokens 与季度切片
    df["_q"] = df["quarter"].astype(str)
    df = df[df["_q"].notna()].copy()
    df = df.sort_values("_q", key=lambda s: s.map(lambda q: qkey(q))).reset_index(drop=True)

    # 全量 tokens（LDA 用）
    tokens_all = texts_to_tokens(df["text_norm"].tolist())

    # 按季度 tokens（DTM 用）
    by_q: Dict[str, List[List[str]]] = {}
    for q, sub in df.groupby("_q"):
        by_q[q] = texts_to_tokens(sub["text_norm"].tolist())

    # --- LDA ---
    # try:
    #     run_lda_k6(tokens_all)
    # except Exception as e:
    #     log_write("[LDA][ERROR] " + repr(e))
    #     log_write(traceback.format_exc())

    # --- DTM ---
    run_dtm_k6_gensim(df, by_q)
    log_write("[DONE] 主题建模完成。输出目录：{}".format(OUT_DIR.resolve()))


if __name__ == "__main__":
    main()