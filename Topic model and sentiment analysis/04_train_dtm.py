# -*- coding: utf-8 -*-train dtm


import os
import re
import time
import logging as log
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as psg
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import ldaseqmodel
import matplotlib.pyplot as plt

# ------------------ 路径与参数 ------------------
INPUT_CSV       = Path("weibo_ai_2023_2025_final_clean.csv")
STOPWORDS_TXT   = Path("baidu_and_hit_stopwords.txt")
USER_DICT_TXT   = Path("userdict.txt")

TEXT_CAND_COLS  = ["内容","文本","text","微博正文","content","full_text"]
TIME_CAND_COLS  = ["时间","发布时间","微博时间","created_at","publish_time",
                   "time","date","日期","发布时间(北京时间)"]

K                = 6
TOPN_SHOW        = 15
MIN_TOKEN_LEN    = 2
MIN_TOKENS_PER_DOC = 3
NO_BELOW         = 10
NO_ABOVE         = 0.40
RANDOM_STATE     = 1

# ------------------ 清洗/时间解析（与你之前一致） ------------------
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

def parse_time_series(s: pd.Series) -> pd.Series:
    def _norm(x: str) -> str:
        if not isinstance(x, str): x = str(x)
        x = x.strip()
        if not x: return x
        if re.fullmatch(r"\d{13}", x):
            try: return pd.to_datetime(int(x), unit="ms").isoformat()
            except: pass
        if re.fullmatch(r"\d{10}", x):
            try: return pd.to_datetime(int(x), unit="s").isoformat()
            except: pass
        x = x.replace("年","-").replace("月","-").replace("日"," ")
        x = x.replace("/", "-").replace(".", "-")
        x = re.sub(r"\s+"," ", x).strip()
        return x
    s2 = s.astype(str).map(_norm)
    return pd.to_datetime(s2, errors="coerce", infer_datetime_format=True)

# ------------------ 分词配置 ------------------
EXTRA_AI_TERMS = [
    "人工智能","生成式人工智能","大模型","小样本学习","对齐","指令微调",
    "大语言模型","语言模型","多模态","Transformers","Transformer",
    "注意力机制","自注意力","预训练","微调","蒸馏","对比学习",
    "Prompt","Prompting","Chain-of-Thought","CoT","RAG","检索增强",
    "知识蒸馏","参数高效微调","LoRA","QLoRA",
    "GPT","GPT4","GPT-4","GPT-4o","ChatGPT",
    "Claude","Llama","Llama2","Llama3","Mixtral","Mistral",
    "DeepSeek","DeepSeek-R1","R1","Qwen","Yi","InternLM","百川","智谱",
    "Sora","Diffusion","扩散模型","VAE","StableDiffusion","SDXL",
    "RLHF","DPO","BPO","PPO","奖励模型","偏好优化",
    "Tokenizer","BPE","SentencePiece","向量数据库","向量检索",
    "多智能体","Agent","Agentic"
]

def load_stopwords(path: Path) -> set:
    if not path.exists():
        base = set()
        log.warning("停用词表不存在：%s（仅用内置补充停用词）", path)
    else:
        base = set(w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())
    extra = {'，', ',', ' ', '。','！','？','：','；','、','（','）','《','》','——','…','～',
             '的','了','和','在','就','都','而','及','与','并','或','一个','我们','你们','他们','它们',
             '这个','那个','这些','那些','啊','呢','嘛','很','还','上','下','中','对','把','给','及其',
             '转发','评论','点赞','链接','网页','图片','视频','直播','秒','图','博主','超话','话题','收起',
             '今天','昨天','明天','年','月','日','时'}
    return base | extra

def load_user_dict(path: Path):
    if path.exists():
        jieba.load_userdict(str(path))
        log.info("已加载自定义词典：%s", path)
    else:
        log.warning("未找到自定义词典：%s（跳过）", path)
    for term in EXTRA_AI_TERMS:
        try:
            tag = "eng" if re.search(r'[A-Za-z]', term) else "nz"
            jieba.add_word(term, freq=10000, tag=tag)
        except Exception:
            pass

def tokenize(text: str, stopwords: set):
    ct = clean_text_func(text)
    if not ct: return []
    toks = []
    for wp in psg.cut(ct):
        w = wp.word.strip()
        if not w: continue
        if len(w) < MIN_TOKEN_LEN: continue
        if w in stopwords: continue
        if w.isdigit(): continue
        toks.append(w)
    if len(toks) < MIN_TOKENS_PER_DOC: return []
    return toks

# ------------------ 工具 ------------------
def quarter_label(dt: pd.Timestamp) -> str:
    y = dt.year
    q = (dt.month - 1)//3 + 1
    return f"{y}Q{q}"

def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CAND_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"未找到文本列；现有列：{list(df.columns)}")

def pick_time_col(df: pd.DataFrame) -> str:
    best_col, best_ok = None, 0
    for c in TIME_CAND_COLS:
        if c in df.columns:
            ok = parse_time_series(df[c]).notna().sum()
            if ok > best_ok:
                best_ok, best_col = ok, c
    if best_col is None or best_ok == 0:
        raise ValueError("未找到可解析的时间列；请确认原始 CSV 的时间列命名与格式。")
    return best_col

# ------------------ 主流程 ------------------
def main():
    log.basicConfig(level=log.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    t0 = time.time()

    assert INPUT_CSV.exists(), f"未找到输入文件：{INPUT_CSV}"
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    log.info("读取CSV shape=%s | 列名(前12)=%s", df.shape, list(df.columns)[:12])

    text_col = pick_text_col(df)
    time_col = pick_time_col(df)
    log.info("检测到文本列：%s | 时间列：%s", text_col, time_col)

    # 时间解析
    dt = parse_time_series(df[time_col])
    ok_mask = dt.notna()
    log.info("[align] 成功匹配时间：%d/%d", ok_mask.sum(), len(df))
    df = df.loc[ok_mask].copy()
    df["_dt"] = dt.loc[ok_mask]
    df["_q"]  = df["_dt"].map(quarter_label)

    # 分词（不做上限采样）
    stopwords = load_stopwords(STOPWORDS_TXT)
    load_user_dict(USER_DICT_TXT)
    texts, quarters = [], []
    for _, row in df.iterrows():
        toks = tokenize(row[text_col], stopwords)
        if toks:
            texts.append(toks)
            quarters.append(row["_q"])
    log.info("有效文档数：%d", len(texts))

    # 以时间先后得到季度顺序
    uniq_q = sorted(set(quarters), key=lambda x: (int(x[:4]), int(x[-1])))
    order_q = uniq_q
    q_counts = defaultdict(int)
    for q in quarters: q_counts[q] += 1
    time_slice = [q_counts[q] for q in order_q]
    log.info("[plan] 季度序列：%s", order_q)
    log.info("[diag] time_slice：%s（总计 %d）", time_slice, sum(time_slice))

    # 词典与语料
    dictionary = corpora.Dictionary(texts)
    log.info("原始词典规模：%d", len(dictionary))
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    dictionary.compactify()
    log.info("[dict] filter_extremes(no_below=%d, no_above=%.2f) → 词典规模：%d",
             NO_BELOW, NO_ABOVE, len(dictionary))

    corpus_all = [dictionary.doc2bow(t) for t in texts]

    # ===== 1) 先用 LDA 初始化（可保留，便于对比/导出；DTM 不再强依赖它）=====
    log.info("[LDA-init] 开始")
    lda_init = LdaMulticore(
        corpus=corpus_all,
        id2word=dictionary,
        num_topics=K,
        workers=max(1, os.cpu_count()//2),
        passes=5,
        iterations=120,
        random_state=RANDOM_STATE,
        chunksize=4000
    )
    lda_init.save("DTM_init_lda.model")
    log.info("[LDA-init] 完成并已保存")

    # ===== 2) 训练 DTM —— 关键修正：不再传 initialize/lda_model，避免 sstats 报错 =====
    log.info("[DTM] 训练开始")
    ldaseq = ldaseqmodel.LdaSeqModel(
        corpus=corpus_all,
        time_slice=time_slice,
        id2word=dictionary,
        num_topics=K,
        lda_inference_max_iter=35,  # 原本 var_max_iter；正确名
        em_min_iter=0,
        em_max_iter=6,
        chunksize=2000,
        random_state=RANDOM_STATE,
        passes=1   # 可调；增大更稳但更慢
    )
    log.info("[DTM] 训练完成，用时 %.1fs", time.time() - t0)

    # ===== 3) 主题随时间演化矩阵 =====
    all_rows = []
    for k in range(K):
        weights = ldaseq.print_topic_times(topic=k)  # 长度与 time_slice 一致
        all_rows.append(weights)
    df_evo = pd.DataFrame({f"Topic{k}": all_rows[k] for k in range(K)}, index=order_q)
    df_evo.to_csv("DTM_topic_times.csv", encoding="utf-8-sig")
    log.info("[save] DTM_topic_times.csv ✅")

    # ===== 5) 每季度×每主题 Top 词到CSV（若 show_topic 可用）=====
    rows = []
    for t, q in enumerate(order_q):
        for k in range(K):
            try:
                pairs = ldaseq.show_topic(topic=k, time=t, topn=TOPN_SHOW)
                terms = " ".join([w for w, _ in pairs])
            except Exception:
                s_list = ldaseq.print_topics(time=t, top_terms=TOPN_SHOW)
                s = s_list[k] if k < len(s_list) else ""
                terms = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5_ ]+", " ", str(s)).strip()
            rows.append({"quarter": q, "topic": k, "top_terms": terms})
    pd.DataFrame(rows).to_csv("DTM_topwords.csv", index=False, encoding="utf-8-sig")
    log.info("[save] DTM_topwords.csv ✅")

    # ===== 6) 保存模型 =====
    import pickle
    with open("my_dtm_model.pkl", "wb") as f:
        pickle.dump(ldaseq, f)
    log.info("[save] my_dtm_model.pkl ✅")

    log.info("[done] 全流程完成，总用时 %.1fs（不做上限）", time.time() - t0)

if __name__ == "__main__":
    main()

