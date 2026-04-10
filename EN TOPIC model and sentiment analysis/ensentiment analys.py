# -*- coding: utf-8 -*-

en

import os
from pathlib import Path
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# ======================
# 0) Paths you may need to modify
# ======================
DTM_MODEL_PATH      = Path("dtm_model.pkl")                   # dtm_model.model
PREPARED_CORPUS_CSV = Path("PREP/prepared_corpus.csv")       
SENTIMENT_PT_PATH   = Path("DL_Cls_Out/bert_lstm_model.pt")  

# output dir
OUT_DIR = Path("topic_sentiment_out_th64_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# inference config
MAX_LEN    = 160
BATCH_SIZE = 32

TEXT_COL_CANDIDATES = ["text_raw", "text_norm", "text", "content", "full_text"]
QUARTER_COL_CANDIDATES = ["quarter", "Quarter", "time", "Time", "slice", "Slice"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold setting
POS_TH = 0.60
NEG_TH = 0.40

# optional font
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================
# Utils: quarter sorting
# ======================
def qkey(q: str):
    s = str(q)
    m = re.match(r"^(20\d{2})Q([1-4])$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), 0)
    m2 = re.match(r"^time(\d+)$", s, re.IGNORECASE)
    if m2:
        return (9999, int(m2.group(1)), 1)
    return (9999, 9999, s)


# ======================
# 1) Load DTM model + gammas
# ======================
def load_dtm_model(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 DTM 模型文件：{path.resolve()}")

    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        dtm = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        return dtm

    try:
        from gensim.models.ldaseqmodel import LdaSeqModel
        dtm = LdaSeqModel.load(str(path))
        return dtm
    except Exception as e:
        raise RuntimeError(f"无法用 gensim 加载 {path.name}：{repr(e)}")


def get_doc_topic_matrix(dtm) -> np.ndarray:
    if hasattr(dtm, "gammas") and dtm.gammas is not None:
        g = dtm.gammas
        if isinstance(g, list):
            gammas = np.vstack(g)
        else:
            gammas = np.asarray(g)
        return gammas

    if hasattr(dtm, "gamma") and dtm.gamma is not None:
        return np.asarray(dtm.gamma)

    raise AttributeError("这个 DTM 对象里找不到 gammas/gamma，无法得到 doc→topic。")


dtm = load_dtm_model(DTM_MODEL_PATH)
K = int(getattr(dtm, "num_topics", 0))
time_slice = list(getattr(dtm, "time_slice", []))

assert K > 0, "DTM 模型里读不到 num_topics"
assert len(time_slice) > 0, "DTM 模型里读不到 time_slice"

gammas = get_doc_topic_matrix(dtm)
N_dtm = gammas.shape[0]

print("✅ DTM loaded:", type(dtm))
print("   K =", K)
print("   time slices =", len(time_slice), " total docs =", sum(time_slice))
print("   gammas shape =", gammas.shape)

if sum(time_slice) != N_dtm:
    raise ValueError(f"❌ time_slice 求和 {sum(time_slice)} != gammas 文档数 {N_dtm}，请检查模型文件。")


# ======================
# 2) Load prepared_corpus and align
# ======================
df = pd.read_csv(PREPARED_CORPUS_CSV)
print("✅ prepared_corpus loaded:", df.shape)

text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is None:
    raise ValueError(
        f"prepared_corpus.csv 找不到文本列。请确保存在以下任意一列：{TEXT_COL_CANDIDATES}\n"
        f"当前列：{list(df.columns)}"
    )

if len(df) != N_dtm:
    raise ValueError(
        f"❌ 行数不一致：prepared_corpus 有 {len(df)} 行，但 DTM gammas 有 {N_dtm} 行。\n"
        f"说明 prepared_corpus 的顺序/范围可能和训练 DTM 时语料不一致。\n"
        f"请使用训练 DTM 时的同一份、同一顺序的表。"
    )

quarter_col = next((c for c in QUARTER_COL_CANDIDATES if c in df.columns), None)

if quarter_col is None:
    ts_path = PREPARED_CORPUS_CSV.parent / "time_slices.csv"
    time_idx = np.concatenate([np.full(n, i) for i, n in enumerate(time_slice)])

    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        tcol = next((c for c in ["time", "Time", "t"] if c in ts.columns), None)
        qcol = next((c for c in ["quarter", "Quarter"] if c in ts.columns), None)
        if tcol and qcol:
            mapping = dict(zip(ts[tcol].tolist(), ts[qcol].astype(str).tolist()))
            df["quarter"] = [mapping.get(int(t), f"time{int(t)}") for t in time_idx]
        else:
            df["quarter"] = [f"time{int(t)}" for t in time_idx]
    else:
        df["quarter"] = [f"time{int(t)}" for t in time_idx]
else:
    df["quarter"] = df[quarter_col].astype(str)

print("✅ quarter prepared.")
print(df[["quarter"]].head())


# ======================
# 3) TRUE topic proportion from gammas
# ======================
topic_cols = [f"Topic{i}" for i in range(K)]

row_sum = gammas.sum(axis=1, keepdims=True).astype(float)
row_sum[row_sum == 0] = 1.0
theta = gammas / row_sum

theta_df = pd.DataFrame(theta, columns=topic_cols)
theta_df["quarter"] = df["quarter"].astype(str).values

dtm_true_times = theta_df.groupby("quarter")[topic_cols].mean()
dtm_true_times = dtm_true_times.div(dtm_true_times.sum(axis=1), axis=0).fillna(0.0)
dtm_true_times = dtm_true_times.loc[sorted(dtm_true_times.index, key=qkey)]

out_true_csv = OUT_DIR / "EN_DTM_topic_times_TRUE_th64_new.csv"
dtm_true_times.reset_index().rename(columns={"quarter": "Quarter"}).to_csv(
    out_true_csv, index=False, encoding="utf-8-sig"
)
print("✅ saved TRUE topic_times:", out_true_csv.resolve())

plt.figure(figsize=(12, 5))
for c in topic_cols:
    plt.plot(dtm_true_times.index, dtm_true_times[c].values, marker="o", label=c)
plt.xticks(rotation=35)
plt.ylabel("Topic intensity")
plt.title("EN DTM Topic Proportion over Time (TRUE) - th64_new")
plt.legend(ncol=3)
plt.tight_layout()
out_line_all = OUT_DIR / "EN_DTM_trend_line_TRUE_th64_new.png"
plt.savefig(out_line_all, dpi=160)
plt.close()
print("✅ saved:", out_line_all.resolve())


# ======================
# 4) Assign topic_id by theta.argmax
# ======================
df["topic_id"] = theta.argmax(axis=1).astype(int)
df["topic"] = df["topic_id"].apply(lambda x: f"Topic{x}")

print("✅ topic prepared.")
print(df[["topic", "quarter"]].head())


# ======================
# 5) Load best sentiment model and predict probability
# ======================
class BertSeqDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [str(x) for x in texts]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class BertLSTMClassifier(nn.Module):
    def __init__(self, model_name, num_classes,
                 lstm_hidden=384, num_layers=1,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state

        lengths = attention_mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        max_len = out_seq.size(1)
        mask = (torch.arange(max_len, device=out_seq.device)[None, :] < lengths[:, None])
        mask = mask.unsqueeze(-1)
        out_seq = out_seq.masked_fill(~mask, -1e9)
        pooled = out_seq.max(dim=1).values

        logits = self.fc(self.dropout(pooled))
        return logits


def load_sentiment_model(pt_path: Path):
    ckpt = torch.load(pt_path, map_location="cpu")
    model_type = ckpt.get("model_type", "")
    if model_type != "bert_lstm":
        print(f"⚠️ 注意：pt 里 model_type={model_type}，当前按 bert_lstm 结构加载。")

    model_name = ckpt["model_name"]
    label2id = ckpt["label2id"]
    id2label = {i: c for c, i in label2id.items()}
    num_classes = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertLSTMClassifier(model_name, num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, tokenizer, label2id, id2label


sent_model, sent_tokenizer, label2id, id2label = load_sentiment_model(SENTIMENT_PT_PATH)

positive_id = None
negative_id = None
for lab, idx in label2id.items():
    if str(lab).lower() == "positive":
        positive_id = idx
    if str(lab).lower() == "negative":
        negative_id = idx

if positive_id is None:
    raise ValueError(f"label2id 中找不到 positive 类，当前：{label2id}")
if negative_id is None:
    print(f"⚠️ label2id 中没显式找到 negative，当前：{label2id}")

ds = BertSeqDataset(df[text_col].tolist(), sent_tokenizer, MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

pred_ids = []
proba_positive = []

with torch.no_grad():
    for batch in dl:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        logits = sent_model(input_ids, attn)

        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        pos_prob = probs[:, positive_id].cpu().numpy().tolist()

        pred_ids.extend(pred)
        proba_positive.extend(pos_prob)

df["sent_pred_argmax"] = [id2label[i] for i in pred_ids]
df["proba_positive"] = proba_positive
df["proba_negative"] = 1 - df["proba_positive"]


def assign_sentiment_by_threshold(p):
    if p > POS_TH:
        return "positive"
    elif p < NEG_TH:
        return "negative"
    else:
        return "uncertain"


df["sent_pred_th64"] = df["proba_positive"].apply(assign_sentiment_by_threshold)

print("✅ threshold sentiment predicted:")
print(df["sent_pred_th64"].value_counts(dropna=False).to_dict())

df_conf = df[df["sent_pred_th64"].isin(["positive", "negative"])].copy()

print("\n✅ confident-only label distribution:")
print(df_conf["sent_pred_th64"].value_counts(dropna=False).to_dict())
print(f"保留样本数: {len(df_conf)} / {len(df)} = {len(df_conf)/len(df):.3f}")


# ======================
# 6) Save tables
# ======================
tab_overall_th64 = (
    df_conf.groupby(["topic", "sent_pred_th64"])
           .size()
           .reset_index(name="count")
)

tab_q_th64 = (
    df_conf.groupby(["quarter", "topic", "sent_pred_th64"])
           .size()
           .reset_index(name="count")
)

out_overall_csv = OUT_DIR / "topic_sentiment_overall_th64_new.csv"
out_byq_csv     = OUT_DIR / "topic_sentiment_by_quarter_th64_new.csv"
out_detail_csv  = OUT_DIR / "prepared_with_topic_and_sentiment_th64_new.csv"
out_detail_all_csv = OUT_DIR / "prepared_with_topic_and_sentiment_all_with_uncertain_th64_new.csv"

tab_overall_th64.to_csv(out_overall_csv, index=False, encoding="utf-8-sig")
tab_q_th64.to_csv(out_byq_csv, index=False, encoding="utf-8-sig")
df_conf.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")
df.to_csv(out_detail_all_csv, index=False, encoding="utf-8-sig")

summary_stats = pd.DataFrame({
    "metric": ["total_docs", "confident_docs", "uncertain_docs", "confident_ratio", "uncertain_ratio"],
    "value": [
        len(df),
        len(df_conf),
        len(df) - len(df_conf),
        round(len(df_conf) / len(df), 6),
        round((len(df) - len(df_conf)) / len(df), 6),
    ]
})
out_stats_csv = OUT_DIR / "sentiment_threshold64_summary_new.csv"
summary_stats.to_csv(out_stats_csv, index=False, encoding="utf-8-sig")

print("✅ saved tables:")
print(" -", out_overall_csv.resolve())
print(" -", out_byq_csv.resolve())
print(" -", out_detail_csv.resolve())
print(" -", out_detail_all_csv.resolve())
print(" -", out_stats_csv.resolve())


# ======================
# 7) Plot overlay for each topic
# ======================
# ======================
# 7) Plot overlay for each topic
# ======================
def plot_topic_overlay_one(
    tab_q: pd.DataFrame,
    dtm_true_times: pd.DataFrame,
    topic_name: str,
    out_png: Path,
    title_prefix: str = "Topic × Sentiment Over Time"
):
    quarters = sorted(
        sorted(
            set(tab_q["quarter"].astype(str).tolist()) |
            set(dtm_true_times.index.astype(str).tolist())
        ),
        key=qkey
    )

    sub = tab_q[tab_q["topic"] == topic_name].copy()
    if sub.empty:
        print(f"⚠️ 跳过：{topic_name} 在 tab_q 里没有数据")
        return

    pivot = sub.pivot_table(
        index="quarter",
        columns="sent_pred_th64",
        values="count",
        aggfunc="sum",
        fill_value=0
    )
    pivot = pivot.reindex(quarters, fill_value=0)

    # 强制柱子顺序：左 positive，右 negative
    desired_cols = ["positive", "negative"]
    for c in desired_cols:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[desired_cols]

    if topic_name not in dtm_true_times.columns:
        raise ValueError(f"dtm_true_times 里没有列 {topic_name}，现有列：{list(dtm_true_times.columns)}")

    line_y = dtm_true_times.reindex(quarters).fillna(0.0)[topic_name].values

    color_map = {
        "positive": "#2A9D8F",
        "negative": "#B56576",
    }
    line_color = "#264653"

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(quarters))
    width = 0.35

    # 左正
    ax.bar(
        x - width/2,
        pivot["positive"].values,
        width=width,
        label="positive (count)",
        color=color_map["positive"],
        alpha=0.85
    )

    # 右负
    ax.bar(
        x + width/2,
        pivot["negative"].values,
        width=width,
        label="negative (count)",
        color=color_map["negative"],
        alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=35, fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylabel("Count", fontsize=20)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        line_y,
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=line_color,
        label="Topic intensity"
    )
    ax2.set_ylabel("Topic intensity", fontsize=20)
    ax2.tick_params(axis='y', labelsize=15)

    en_topic_num = int(topic_name.replace("Topic", ""))
    ax.set_title(
        f"{title_prefix} — EN topic{en_topic_num}",
        fontsize=20,
        pad=10
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=True,
        fontsize=11
    )

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_topics_overlay(tab_q, dtm_true_times, out_dir: Path):
    for i in range(K):
        t = f"Topic{i}"
        out_png = out_dir / f"topic_sentiment_over_time_{t}_th64_new.png"
        plot_topic_overlay_one(tab_q, dtm_true_times, t, out_png)


plot_all_topics_overlay(tab_q_th64, dtm_true_times, OUT_DIR)

print("🎉 DONE. 输出目录：", OUT_DIR.resolve())
print("关键输出：")
print(" - TRUE topic_times:", out_true_csv.resolve())
print(" - by quarter (th64):", out_byq_csv.resolve())
print(" - summary:", out_stats_csv.resolve())
print(" - example overlay:", (OUT_DIR / "topic_sentiment_over_time_Topic0_th64_new.png").resolve())
