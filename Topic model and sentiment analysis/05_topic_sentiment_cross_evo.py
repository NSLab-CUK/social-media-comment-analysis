# -*- coding: utf-8 -*-得到情感分析文件

#topic_sentiment_cross_evo.py
#目标：主题×情感交叉演化（季度×主题 -> 平均情感/正向比例/数量）

#依赖文件：
#- weibo_ai_2023_2025_final_clean.csv
#- my_dtm_model.gensim 或 my_dtm_model.pkl
#- DL_Cls_Out/artifacts/bert_cls_minimal/best  （你的微调情感模型）
#- (可选) DTM_topic_labels.csv  （topic->最终6类映射）

#输出：
#- doc_topic_sentiment.csv
#- quarter_topic_sentiment.csv

import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jieba
import jieba.posseg as psg

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models.ldaseqmodel import LdaSeqModel


# ========== 路径 ==========
INPUT_CSV = Path("weibo_ai_2023_2025_final_clean.csv")

DTM_GENSIM = Path("my_dtm_model.gensim")


SENT_DIR   = Path("DL_Cls_Out/artifacts/bert_cls_minimal/best")

LABEL_MAP_CSV = Path("DTM_topic_labels.csv")  # 可选

# ========== 和你DTM保持一致的关键参数（按你给的脚本） ==========
TEXT_CAND_COLS  = ["内容","文本","text","微博正文","content","full_text"]
TIME_CAND_COLS  = ["时间","发布时间","微博时间","created_at","publish_time",
                   "time","date","日期","发布时间(北京时间)"]

K = 6
MIN_TOKEN_LEN = 2
MIN_TOKENS_PER_DOC = 3

STOPWORDS_TXT = Path("baidu_and_hit_stopwords.txt")
USER_DICT_TXT = Path("userdict.txt")


# ========== 清洗 / 时间解析（照你DTM脚本） ==========
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
        raise ValueError("未找到可解析的时间列。")
    return best_col

def load_stopwords(path: Path) -> set:
    base = set()
    if path.exists():
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

def tokenize_for_filter(text: str, stopwords: set):
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
    if len(toks) < MIN_TOKENS_PER_DOC:
        return []
    return toks


# ========== 1) 复现“DTM有效文档序列”（关键：顺序要一致） ==========
def build_valid_docs_df():
    assert INPUT_CSV.exists(), f"缺少 {INPUT_CSV}"
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    text_col = pick_text_col(df)
    time_col = pick_time_col(df)

    dt = parse_time_series(df[time_col])
    ok = dt.notna()
    df = df.loc[ok].copy()
    df["_dt"] = dt.loc[ok]
    df["quarter"] = df["_dt"].map(quarter_label)

    # 取一个ID列（你截图里有 ID；如果没有就用行号兜底）
    id_col = "ID" if "ID" in df.columns else None

    stopwords = load_stopwords(STOPWORDS_TXT)
    load_user_dict(USER_DICT_TXT)

    rows = []
    for idx, r in df.iterrows():
        toks = tokenize_for_filter(r[text_col], stopwords)
        if toks:
            rows.append({
                "doc_index_raw": int(idx),
                "weibo_id": str(r[id_col]) if id_col else str(idx),
                "quarter": r["quarter"],
                "text": str(r[text_col]),
                "text_clean": clean_text_func(str(r[text_col])),
            })
    out = pd.DataFrame(rows)
    return out


# ========== 2) 加载DTM模型，并从 gammas 得到每篇微博主导主题 ==========
def load_dtm():
    if DTM_GENSIM.exists():
        return LdaSeqModel.load(str(DTM_GENSIM))
    if DTM_PKL.exists():
        with open(DTM_PKL, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("没找到 my_dtm_model.gensim / my_dtm_model.pkl")

def get_dom_topic_from_gammas(ldaseq: LdaSeqModel, n: int):
    gammas = np.asarray(ldaseq.gammas, dtype=float)  # shape: (n_docs, K)
    n_model = gammas.shape[0]
    if n > n_model:
        # 保险：截断，避免越界
        n = n_model
    denom = gammas[:n].sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    theta = gammas[:n] / denom
    dom = theta.argmax(axis=1).astype(int)
    # 如果你想保留完整分布也可以返回 theta
    return dom, n


# ========== 3) 用你微调好的BERT对每条微博做情感推理 ==========
@torch.no_grad()
def predict_sentiment(texts, model_dir: Path, batch_size=64, max_len=160):
    assert model_dir.exists(), f"没找到情感模型目录：{model_dir}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p_pos = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p_pos.tolist())

    probs = np.array(probs, dtype=float)
    pred = (probs >= 0.5).astype(int)
    return probs, pred


# ========== 4) topic->label 映射（可读CSV；没有就用默认） ==========
def load_topic_label_map():
    default = {
        0: "海外大厂动向",
        1: "技术应用进展",
        2: "人机伦理思考",
        3: "产业发展政策",
        4: "A股市场风向",
        5: "教育科研推广",
    }
    if LABEL_MAP_CSV.exists():
        mdf = pd.read_csv(LABEL_MAP_CSV, encoding="utf-8-sig")
        if "topic" in mdf.columns and "label" in mdf.columns:
            return {int(r["topic"]): str(r["label"]) for _, r in mdf.iterrows()}
    return default


def main():
    # A) 复现DTM有效文档序列
    dfv = build_valid_docs_df()
    print("[info] valid docs =", len(dfv))

    # B) DTM主题（从gammas取主导主题）
    ldaseq = load_dtm()
    dom_topic, n_used = get_dom_topic_from_gammas(ldaseq, len(dfv))
    dfv = dfv.iloc[:n_used].copy()
    dfv["topic_id"] = dom_topic

    # C) 情感推理（用微调好的BERT）
    probs, pred = predict_sentiment(dfv["text_clean"].tolist(), SENT_DIR, batch_size=64, max_len=160)
    dfv["sent_pos_prob"] = probs
    dfv["sent_label"] = pred

    # D) topic->最终6类标签
    tmap = load_topic_label_map()
    dfv["topic_label"] = dfv["topic_id"].map(lambda x: tmap.get(int(x), f"Topic{x}"))

    # E) 导出逐微博文件
    dfv_out = dfv[["weibo_id", "quarter", "topic_id", "topic_label", "sent_pos_prob", "sent_label", "text_clean"]].copy()
    dfv_out.to_csv("doc_topic_sentiment.csv", index=False, encoding="utf-8-sig")
    print("[save] doc_topic_sentiment.csv")

    # F) 聚合：季度×主题
    agg = (dfv_out
           .groupby(["quarter", "topic_id", "topic_label"], as_index=False)
           .agg(
               n_docs=("weibo_id", "count"),
               mean_sent=("sent_pos_prob", "mean"),
               pos_rate=("sent_label", "mean"),
           ))

    # 排序季度
    def _qkey(q):
        m = re.search(r"(\d{4})Q([1-4])", str(q))
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)
    agg = agg.sort_values(["quarter", "topic_id"], key=lambda s: s.map(_qkey) if s.name=="quarter" else s)
    agg.to_csv("quarter_topic_sentiment.csv", index=False, encoding="utf-8-sig")
    print("[save] quarter_topic_sentiment.csv")

    print("✅ done")

if __name__ == "__main__":
    main()
