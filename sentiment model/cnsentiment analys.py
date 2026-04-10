# -*- coding: utf-8 -*-cn

import re
import pickle
import ast
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models.ldaseqmodel import LdaSeqModel


# ========== 路径 ==========
RAW_INPUT_CSV = Path("weibo_ai_2023_2025_final_clean.csv")
TOKENS_CSV    = Path("weibo_ai_tokens_2023_2025.csv")

DTM_GENSIM = Path("my_dtm_model.gensim")
DTM_PKL    = Path("my_dtm_model.pkl")

SENT_DIR = Path("DL_Cls_Out/artifacts/bert_cls_minimal/best")
LABEL_MAP_CSV = Path("DTM_topic_labels.csv")   # 可选

# ========== 输出 ==========
DOC_ALL_OUT  = Path("doc_topic_sentiment_all_with_uncertain_th64_tokens_align.csv")
DOC_CONF_OUT = Path("doc_topic_sentiment_th64_tokens_align.csv")
QUARTER_OUT  = Path("quarter_topic_sentiment_th64_tokens_align.csv")
SUMMARY_OUT  = Path("sentiment_threshold64_summary_cn_tokens_align.csv")

# ========== 阈值 ==========
POS_TH = 0.60
NEG_TH = 0.40

# ========== 对齐口径 ==========
TEXT_CAND = ["内容", "文本", "text", "微博正文", "content", "full_text"]
FORCE_DT_COL = "时间"     # 和第二个脚本一致
MIN_TOKENS_PER_DOC = 2    # 和第二个脚本一致


# ========== 文本清洗 ==========
_re_url   = re.compile(r'https?://\S+|www\.\S+')
_re_at    = re.compile(r'@[\w\-\u4e00-\u9fff]+')
_re_topic = re.compile(r'#([^#]+)#')
_re_space = re.compile(r'\s+')
_re_keep  = re.compile(r'[A-Za-z0-9\u4e00-\u9fa5]+')

def clean_text_func(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _re_url.sub(" ", s)
    s = _re_at.sub(" ", s)
    s = _re_topic.sub(" ", s)
    s = _re_space.sub(" ", s).strip()
    parts = _re_keep.findall(s)
    return " ".join(parts)

def find_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CAND:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == "O"]
    if obj:
        lens = {c: df[c].astype(str).str.len().mean() for c in obj}
        return max(lens, key=lens.get)
    raise ValueError("未找到文本列")

def _normalize_time_str(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return s

    if re.fullmatch(r"\d{13}", s):
        try:
            return pd.to_datetime(int(s), unit="ms").isoformat()
        except:
            pass

    if re.fullmatch(r"\d{10}", s):
        try:
            return pd.to_datetime(int(s), unit="s").isoformat()
        except:
            pass

    s = s.replace("年", "-").replace("月", "-").replace("日", " ")
    s = s.replace("/", "-").replace(".", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s

COMMON_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%Z", "%Y-%m-%dT%H:%M:%S",
]

def parse_time_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).map(_normalize_time_str)
    dt = pd.to_datetime(s2, errors="coerce", utc=False, infer_datetime_format=True)
    if dt.notna().sum() == 0:
        for fmt in COMMON_FORMATS:
            dt2 = pd.to_datetime(s2, format=fmt, errors="coerce")
            if dt2.notna().sum() > 0:
                return dt2
    return dt

def quarter_label(dt: pd.Timestamp) -> str:
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


# ========== 1) 改成第二个脚本口径：tokens + clean_text 对齐时间 ==========
def build_valid_docs_df():
    assert TOKENS_CSV.exists(), f"缺少 {TOKENS_CSV}"
    assert RAW_INPUT_CSV.exists(), f"缺少 {RAW_INPUT_CSV}"

    # 1) 读取 tokens 表
    df_tok = pd.read_csv(TOKENS_CSV, encoding="utf-8-sig")
    assert "clean_text" in df_tok.columns and "tokens" in df_tok.columns, \
        "tokens CSV 必须包含 clean_text 和 tokens 两列"

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

    print(f"[info] tokens rows = {len(df_tok)}")
    print(f"[info] unique clean_text in tokens = {df_tok['clean_text'].nunique()}")

    # 2) 读取原始表，生成 clean_text + 时间
    df_raw = pd.read_csv(RAW_INPUT_CSV, encoding="utf-8-sig")
    text_col = find_text_col(df_raw)
    df_raw["_clean_text"] = df_raw[text_col].astype(str).map(clean_text_func)

    time_col = FORCE_DT_COL if FORCE_DT_COL in df_raw.columns else None
    if time_col is None:
        raise ValueError(f"原始表中未找到时间列：{FORCE_DT_COL}")

    df_raw["_dt"] = parse_time_series(df_raw[time_col])
    print(f"[time] 使用时间列：{time_col} | 可解析：{df_raw['_dt'].notna().sum()}/{len(df_raw)}")

    # 3) 与第二个脚本一致：原始表仅用于建立 clean_text -> 时间 映射
    df_map = (
        df_raw
        .dropna(subset=["_clean_text"])
        .sort_index()
        .drop_duplicates(subset=["_clean_text"], keep="first")[["_clean_text", "_dt"]]
    )

    # 4) 用 clean_text 合并时间到 tokens 表
    df = df_tok.merge(df_map, left_on="clean_text", right_on="_clean_text", how="left")
    matched = df["_dt"].notna().sum()
    print(f"[align] 按 clean_text 成功匹配时间：{matched}/{len(df)}")

    # 5) 过滤：有时间 + token 数量达标
    df = df[df["_dt"].notna()].copy()
    df["_dt"] = pd.to_datetime(df["_dt"], errors="coerce")
    before_len = len(df)
    df = df[df["tok_len"] >= MIN_TOKENS_PER_DOC].reset_index(drop=True)
    print(f"[filter] token >= {MIN_TOKENS_PER_DOC}: {len(df)}/{before_len}")

    # 6) 构建与后续流程兼容的字段
    if "ID" in df.columns:
        weibo_ids = df["ID"].astype(str)
    else:
        weibo_ids = pd.Series(np.arange(len(df)), index=df.index).astype(str)

    df["quarter"] = df["_dt"].apply(quarter_label)
    df["weibo_id"] = weibo_ids

    # 这里 text 直接用 clean_text，和旧脚本后续情感推理兼容
    out = df[["weibo_id", "quarter", "clean_text"]].copy()
    out = out.rename(columns={"clean_text": "text_clean"})

    print(f"[info] final docs for sentiment/topic = {len(out)}")
    return out


# ========== 2) 加载 DTM，并提取主主题 ==========
def load_dtm():
    if DTM_GENSIM.exists():
        return LdaSeqModel.load(str(DTM_GENSIM))
    if DTM_PKL.exists():
        with open(DTM_PKL, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "model" in obj:
            return obj["model"]
        return obj
    raise FileNotFoundError("没找到 my_dtm_model.gensim / my_dtm_model.pkl")

def get_dom_topic_from_gammas(ldaseq, n: int):
    gammas = np.asarray(ldaseq.gammas, dtype=float)
    n_model = gammas.shape[0]
    if n > n_model:
        print(f"[warn] docs={n} > model_docs={n_model}，将截断到 {n_model}")
        n = n_model

    denom = gammas[:n].sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    theta = gammas[:n] / denom
    dom = theta.argmax(axis=1).astype(int)
    return dom, theta, n


# ========== 3) 情感预测 ==========
@torch.no_grad()
def predict_sentiment_probs(texts, model_dir: Path, batch_size=64, max_len=160):
    assert model_dir.exists(), f"没找到情感模型目录：{model_dir}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        p_pos = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p_pos.tolist())

    return np.array(probs, dtype=float)

def assign_sentiment_by_threshold(p):
    if p > POS_TH:
        return "positive"
    elif p < NEG_TH:
        return "negative"
    else:
        return "uncertain"


# ========== 4) topic 标签 ==========
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

def qkey(q):
    m = re.search(r"(\d{4})Q([1-4])", str(q))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)


# ========== 主流程 ==========
def main():
    # A) 改成第二个脚本口径
    dfv = build_valid_docs_df()
    print("[info] valid docs =", len(dfv))

    # B) DTM 主主题
    ldaseq = load_dtm()
    dom_topic, theta, n_used = get_dom_topic_from_gammas(ldaseq, len(dfv))

    dfv = dfv.iloc[:n_used].copy()
    dfv["topic_id"] = dom_topic
    dfv["topic_strength"] = theta[np.arange(n_used), dom_topic]

    # C) 情感推理
    probs = predict_sentiment_probs(
        dfv["text_clean"].tolist(),
        SENT_DIR,
        batch_size=64,
        max_len=160
    )
    dfv["sent_pos_prob"] = probs
    dfv["sent_neg_prob"] = 1.0 - dfv["sent_pos_prob"]
    dfv["sent_pred"] = dfv["sent_pos_prob"].apply(assign_sentiment_by_threshold)
    dfv["sent_pred_argmax"] = np.where(dfv["sent_pos_prob"] >= 0.5, "positive", "negative")

    # D) topic label
    tmap = load_topic_label_map()
    dfv["topic_label"] = dfv["topic_id"].map(lambda x: tmap.get(int(x), f"Topic{x}"))

    # E) 导出全量
    dfv_all_out = dfv[
        ["weibo_id", "quarter", "topic_id", "topic_label", "topic_strength",
         "sent_pos_prob", "sent_neg_prob", "sent_pred", "sent_pred_argmax", "text_clean"]
    ].copy()
    dfv_all_out.to_csv(DOC_ALL_OUT, index=False, encoding="utf-8-sig")
    print(f"[save] {DOC_ALL_OUT}")

    # F) 高置信度
    dfv_conf = dfv[dfv["sent_pred"].isin(["positive", "negative"])].copy()
    dfv_conf_out = dfv_conf[
        ["weibo_id", "quarter", "topic_id", "topic_label", "topic_strength",
         "sent_pos_prob", "sent_neg_prob", "sent_pred", "text_clean"]
    ].copy()
    dfv_conf_out.to_csv(DOC_CONF_OUT, index=False, encoding="utf-8-sig")
    print(f"[save] {DOC_CONF_OUT}")

    # G) 阈值统计
    n_total = len(dfv)
    n_conf = len(dfv_conf)
    n_unc = n_total - n_conf

    summary = pd.DataFrame({
        "metric": [
            "total_docs",
            "confident_docs",
            "uncertain_docs",
            "confident_ratio",
            "uncertain_ratio",
            "positive_docs_confident",
            "negative_docs_confident"
        ],
        "value": [
            n_total,
            n_conf,
            n_unc,
            round(n_conf / n_total, 6) if n_total else 0,
            round(n_unc / n_total, 6) if n_total else 0,
            int((dfv_conf["sent_pred"] == "positive").sum()),
            int((dfv_conf["sent_pred"] == "negative").sum()),
        ]
    })
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    print(f"[save] {SUMMARY_OUT}")

    print("\n[info] sent_pred distribution (all):")
    print(dfv["sent_pred"].value_counts(dropna=False).to_dict())

    print("\n[info] sent_pred distribution (confident only):")
    print(dfv_conf["sent_pred"].value_counts(dropna=False).to_dict())

    # H) 季度 × 主题聚合
    agg = (
        dfv_conf
        .groupby(["quarter", "topic_id", "topic_label"], as_index=False)
        .agg(
            n_docs=("weibo_id", "count"),
            mean_sent=("sent_pos_prob", "mean"),
            pos_rate=("sent_pred", lambda x: (x == "positive").mean()),
            neg_rate=("sent_pred", lambda x: (x == "negative").mean()),
            pos_count=("sent_pred", lambda x: (x == "positive").sum()),
            neg_count=("sent_pred", lambda x: (x == "negative").sum()),
        )
    )

    agg = agg.sort_values(
        ["quarter", "topic_id"],
        key=lambda s: s.map(qkey) if s.name == "quarter" else s
    )
    agg.to_csv(QUARTER_OUT, index=False, encoding="utf-8-sig")
    print(f"[save] {QUARTER_OUT}")

    print("\n✅ done")


if __name__ == "__main__":
    main()


# ======== Part 2: Aggregation and plotting ========

# -*- coding: utf-8 -*-
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================
# 0) 路径设置
# ======================
DTM_MODEL_PATH      = Path("my_dtm_model.gensim")
PREPARED_CORPUS_CSV = Path("weibo_ai_dtm_train_subset_47137.csv")
DOC_SENT_CSV        = Path("doc_topic_sentiment_all_with_uncertain_th64_tokens_align.csv")

OUT_DIR = Path("cn_topic_sentiment_out_th64_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 主题命名
TOPIC_TO_FINAL = {
    0: "海外大厂动向",
    1: "技术应用进展",
    2: "人机伦理思考",
    3: "产业发展政策",
    4: "A股市场风向",
    5: "教育科研推广",
}

# 颜色
C_POS  = "#2A9D8F"
C_NEG  = "#B56576"
C_LINE = "#264653"

# 字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Songti SC", "STHeiti", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# prepared_corpus 可能列名
TEXT_COL_CANDIDATES = ["text_raw", "text_norm", "text", "content", "full_text"]
QUARTER_COL_CANDIDATES = ["quarter", "Quarter", "time", "Time", "slice", "Slice"]
ID_COL_CANDIDATES = ["ID", "id", "weibo_id", "doc_id"]

# 情感文件可能列名
SENT_COL_CANDIDATES = ["sent_pred", "sent_label", "sentiment", "pred", "label"]

# 文本清洗（给 text_clean 对齐用）
_re_url   = re.compile(r'https?://\S+|www\.\S+')
_re_at    = re.compile(r'@[\w\-\u4e00-\u9fff]+')
_re_topic = re.compile(r'#([^#]+)#')
_re_space = re.compile(r'\s+')
_re_keep  = re.compile(r'[A-Za-z0-9\u4e00-\u9fa5]+')


# ======================
# 1) 工具函数
# ======================
def clean_text_func(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _re_url.sub(" ", s)
    s = _re_at.sub(" ", s)
    s = _re_topic.sub(" ", s)
    s = _re_space.sub(" ", s).strip()
    parts = _re_keep.findall(s)
    return " ".join(parts)

def normalize_sent_label(x):
    s = str(x).strip().lower()
    if s in ("1", "pos", "positive", "true", "yes"):
        return "positive"
    if s in ("0", "neg", "negative", "false", "no"):
        return "negative"
    if s in ("uncertain",):
        return "uncertain"
    if s in ("positive", "negative"):
        return s
    return np.nan

def overlay_png(k: int, label: str) -> Path:
    safe = str(label).replace("/", "_").replace("\\", "_").replace(" ", "")
    return OUT_DIR / f"{k:02d}_{safe}_overlay_th64_final.png"

def load_ldaseq_model(path: Path):
    path = Path(path)
    assert path.exists(), f"找不到 DTM 模型：{path.resolve()}"
    from gensim.models.ldaseqmodel import LdaSeqModel
    return LdaSeqModel.load(str(path))


# ======================
# 2) 加载 DTM
# ======================
dtm = load_ldaseq_model(DTM_MODEL_PATH)
K = int(dtm.num_topics)
T = int(dtm.num_time_slices)
time_slice = list(dtm.time_slice)
gammas = np.asarray(dtm.gammas, dtype=float)

print("✅ DTM loaded:", type(dtm))
print("   K =", K, "| T =", T)
print("   time slices =", len(time_slice), " total docs =", sum(time_slice))
print("   gammas shape =", gammas.shape)

assert K == 6, f"你现在 K={K}，不是6的话请检查模型"
assert len(time_slice) == T
assert sum(time_slice) == gammas.shape[0]

N = gammas.shape[0]


# ======================
# 3) 读取 prepared_corpus
# ======================
assert PREPARED_CORPUS_CSV.exists(), f"找不到：{PREPARED_CORPUS_CSV.resolve()}"
df = pd.read_csv(PREPARED_CORPUS_CSV)
print("✅ prepared_corpus loaded:", df.shape)

if len(df) != N:
    raise ValueError(
        f"❌ 行数不一致：prepared_corpus={len(df)} vs DTM docs={N}\n"
        f"必须使用训练 DTM 时同一份、同一顺序的 prepared_corpus.csv"
    )

quarter_col = next((c for c in QUARTER_COL_CANDIDATES if c in df.columns), None)
if quarter_col is None:
    time_idx = np.concatenate([np.full(n, i) for i, n in enumerate(time_slice)])
    df["quarter"] = [f"time{int(t)}" for t in time_idx]
    quarter_col = "quarter"

df["_quarter_clean"] = (
    df[quarter_col].astype(str)
      .str.upper()
      .str.replace(r"\s+", "", regex=True)
)

# time_slice 对齐 quarter
quarters_use = []
s = 0
for i, n in enumerate(time_slice):
    block_q = df["_quarter_clean"].iloc[s:s+n]
    if len(block_q) == 0:
        quarters_use.append(f"time{i}")
    else:
        quarters_use.append(block_q.value_counts().idxmax())
    s += n

print("✅ quarters (aligned to time_slice):", quarters_use)


# ======================
# 4) 计算 TRUE topic_times
# ======================
topic_times = []
s = 0
for n in time_slice:
    block = gammas[s:s+n]

    rs = block.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    theta = block / rs

    m = theta.mean(axis=0)
    m = m / (m.sum() if m.sum() != 0 else 1.0)

    topic_times.append(m)
    s += n

topic_times = np.vstack(topic_times)
cols = [f"Topic{i}" for i in range(K)]

df_true = pd.DataFrame(topic_times, columns=cols)
df_true.insert(0, "Quarter", quarters_use)

TRUE_CSV = OUT_DIR / "CN_DTM_topic_times_TRUE_th64_final.csv"
df_true.to_csv(TRUE_CSV, index=False, encoding="utf-8-sig")
print("✅ saved TRUE topic_times:", TRUE_CSV.resolve())

# 折线图
plt.figure(figsize=(12, 5))
for c in cols:
    plt.plot(df_true["Quarter"], df_true[c], marker="o", label=c)
plt.xticks(rotation=35, fontsize=20)
plt.ylabel("Topic Intensity", fontsize=20, fontweight="bold")
plt.title("CN DTM Topic Intensity over Time", fontsize=20, fontweight="bold")
plt.yticks(fontsize=15)
plt.legend(ncol=3)
plt.tight_layout()
LINE_PNG = OUT_DIR / "CN_DTM_trend_line_TRUE_th64_final.png"
plt.savefig(LINE_PNG, dpi=160)
plt.close()
print("✅ saved:", LINE_PNG.resolve())

# 堆叠图
plt.figure(figsize=(12, 5))
plt.stackplot(df_true["Quarter"], *[df_true[c].to_numpy(float) for c in cols], labels=cols)
plt.xticks(rotation=35)
plt.ylabel("Proportion (TRUE)")
plt.title("CN DTM Topic Proportion over Time (Stacked, TRUE)_th64_final")
plt.legend(ncol=3, loc="upper center")
plt.tight_layout()
STACK_PNG = OUT_DIR / "CN_DTM_trend_stacked_TRUE_th64_final.png"
plt.savefig(STACK_PNG, dpi=160)
plt.close()
print("✅ saved:", STACK_PNG.resolve())


# ======================
# 5) 准备每条 DTM 文档的信息
# ======================
df_doc = pd.DataFrame({
    "quarter": df["_quarter_clean"].tolist(),
})
df_doc["topic_id"] = gammas.argmax(axis=1).astype(int)
df_doc["topic"] = df_doc["topic_id"].apply(lambda x: f"Topic{x}")

# 准备可匹配键：weibo_id
id_col = next((c for c in ID_COL_CANDIDATES if c in df.columns), None)
if id_col is not None:
    df_doc["weibo_id"] = df[id_col].astype(str)
else:
    df_doc["weibo_id"] = np.nan

# 准备可匹配键：text_clean
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is not None:
    df_doc["text_clean"] = df[text_col].astype(str).map(clean_text_func)
else:
    df_doc["text_clean"] = np.nan


# ======================
# 6) 读取情感结果，尽量匹配
# ======================
assert DOC_SENT_CSV.exists(), f"找不到情感文件：{DOC_SENT_CSV.resolve()}"
ds = pd.read_csv(DOC_SENT_CSV, encoding="utf-8-sig")
print("✅ sentiment file loaded:", ds.shape)

sent_col = next((c for c in SENT_COL_CANDIDATES if c in ds.columns), None)
if sent_col is None:
    raise ValueError("❌ 情感结果文件缺少 sent_pred / sentiment 这类列。")

ds = ds.copy()
ds["sent_pred_norm"] = ds[sent_col].map(normalize_sent_label)

df_doc["sent_pred"] = np.nan
matched_by = None

# 方案1：整行对齐
if len(ds) == N:
    df_doc["sent_pred"] = ds["sent_pred_norm"].tolist()
    matched_by = "row_alignment"

# 方案2：weibo_id 对齐
elif "weibo_id" in ds.columns and df_doc["weibo_id"].notna().any():
    tmp = ds[["weibo_id", "sent_pred_norm"]].copy()
    tmp["weibo_id"] = tmp["weibo_id"].astype(str)
    tmp = tmp.drop_duplicates(subset=["weibo_id"], keep="first")

    merged = df_doc.merge(tmp, on="weibo_id", how="left", suffixes=("", "_y"))
    df_doc["sent_pred"] = merged["sent_pred_norm"]
    matched_by = "weibo_id"

# 方案3：text_clean 对齐
elif "text_clean" in ds.columns and df_doc["text_clean"].notna().any():
    tmp = ds[["text_clean", "sent_pred_norm"]].copy()
    tmp["text_clean"] = tmp["text_clean"].astype(str)
    tmp = tmp.drop_duplicates(subset=["text_clean"], keep="first")

    merged = df_doc.merge(tmp, on="text_clean", how="left", suffixes=("", "_y"))
    df_doc["sent_pred"] = merged["sent_pred_norm"]
    matched_by = "text_clean"

else:
    print("⚠️ 没有找到可用匹配键，情感结果无法贴回 DTM 文档。")

n_matched = df_doc["sent_pred"].notna().sum()
match_ratio = n_matched / N if N else 0

print(f"✅ sentiment matched by: {matched_by}")
print(f"✅ matched sentiment docs: {n_matched}/{N}")
print(f"✅ match ratio: {match_ratio:.2%}")
print("✅ raw sentiment distribution:", df_doc["sent_pred"].value_counts(dropna=False).to_dict())

# 保存对齐后的逐文档结果，便于你检查
DOC_ALIGN_CSV = OUT_DIR / "dtm_docs_with_sentiment_alignment_th64_final.csv"
df_doc.to_csv(DOC_ALIGN_CSV, index=False, encoding="utf-8-sig")
print("✅ saved:", DOC_ALIGN_CSV.resolve())


# ======================
# 7) 聚合 quarter × topic × sentiment
# ======================
df_doc_plot = df_doc[df_doc["sent_pred"].isin(["positive", "negative"])].copy()

tab_q = (
    df_doc_plot.groupby(["quarter", "topic", "sent_pred"])
               .size()
               .reset_index(name="count")
)

TAB_Q_CSV = OUT_DIR / "topic_sentiment_by_quarter_th64_final.csv"
tab_q.to_csv(TAB_Q_CSV, index=False, encoding="utf-8-sig")
print("✅ saved:", TAB_Q_CSV.resolve())


# ======================
# 8) 画每个主题的 overlay 图
# ======================
# ======================
# 8) 画每个主题的 overlay 图
#    画图风格改成和参考脚本一致：
#    - figsize=(12,5)
#    - x轴 fontsize=20
#    - y轴 tick fontsize=15
#    - ylabel fontsize=20
#    - title fontsize=20, pad=10
#    - legend fontsize=11
#    但标题文字保持你原来的不变
#    并固定：左 positive，右 negative
# ======================
quarters_plot = df_true["Quarter"].astype(str).tolist()
true_line = {f"Topic{i}": df_true[f"Topic{i}"].to_numpy(float) for i in range(K)}

for k in range(K):
    topic_name = TOPIC_TO_FINAL.get(k, f"Topic{k}")
    topic_key = f"Topic{k}"

    y_line = true_line[topic_key]

    sub = tab_q[tab_q["topic"] == topic_key].copy()
    pv = sub.pivot(index="quarter", columns="sent_pred", values="count").fillna(0)
    pv = pv.reindex(quarters_plot).fillna(0)

    # 固定顺序：左 positive，右 negative
    pos = pv["positive"].to_numpy(float) if "positive" in pv.columns else np.zeros(len(quarters_plot))
    neg = pv["negative"].to_numpy(float) if "negative" in pv.columns else np.zeros(len(quarters_plot))

    x = np.arange(len(quarters_plot))
    w = 0.35

    # 和参考脚本一致：figsize=(12,5)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 左边 positive，右边 negative
    ax1.bar(
        x - w/2, pos,
        width=w,
        label="positive (count)",
        color=C_POS,
        alpha=0.85
    )
    ax1.bar(
        x + w/2, neg,
        width=w,
        label="negative (count)",
        color=C_NEG,
        alpha=0.85
    )

    # 坐标轴样式：按参考脚本改
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarters_plot, rotation=35, fontsize=20)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.set_ylabel("Count", fontsize=20)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        y_line,
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=C_LINE,
        label="DTM topic intensity"
    )
    ax2.set_ylabel("Topic intensity", fontsize=20)
    ax2.tick_params(axis="y", labelsize=15)

    # 标题文字保持你原来的不变，只改样式
    ax1.set_title(
        f"Topic × Sentiment Over Time — CN topic{k}",
        fontsize=20,
        pad=10
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=True,
        fontsize=11
    )

    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    out_png = overlay_png(k, topic_name)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[debug] {topic_name} TRUE proportions:", np.round(y_line, 6).tolist())
    print("✅ saved:", out_png.resolve())