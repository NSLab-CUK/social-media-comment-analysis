# -*- coding: utf-8 -*-
"""
EN: 用“ DTM topic proportion（从 dtm_model.pkl 的 gammas + time_slice 得到）”
     叠加“topic×sentiment 的计数柱状图”，输出 overlay 图（每个 Topic 一张）。
所有输出文件/目录都加 _new，避免和旧结果混淆。
"""

import os
from pathlib import Path
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# ======================
# 0) 你需要改的 3 个路径
# ======================
DTM_MODEL_PATH      = Path("dtm_model.pkl")                   # 或 dtm_model.model
PREPARED_CORPUS_CSV = Path("PREP/prepared_corpus.csv")        # 必须与 DTM 文档顺序一致
SENTIMENT_PT_PATH   = Path("DL_Cls_Out/bert_lstm_model.pt")   # 你的情感模型

# 输出目录（加 _new）
OUT_DIR = Path("topic_sentiment_out_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 推理配置
MAX_LEN    = 160
BATCH_SIZE = 32

TEXT_COL_CANDIDATES = ["text_raw", "text_norm", "text", "content", "full_text"]
QUARTER_COL_CANDIDATES = ["quarter", "Quarter", "time", "Time", "slice", "Slice"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (可选) 字体（Mac）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================
# 工具：季度排序（支持 2023Q1 / time0）
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
# 1) 读取 DTM 模型 + doc-topic(gammas)
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

    # .model：尝试用 gensim LdaSeqModel.load
    try:
        from gensim.models.ldaseqmodel import LdaSeqModel
        dtm = LdaSeqModel.load(str(path))
        return dtm
    except Exception as e:
        raise RuntimeError(f"无法用 gensim 加载 {path.name}：{repr(e)}")

def get_doc_topic_matrix(dtm) -> np.ndarray:
    """
    返回 (N_docs, K_topics) 的矩阵（gammas）
    """
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
# 2) 读取 prepared_corpus.csv，并对齐行数、补 quarter
# ======================
df = pd.read_csv(PREPARED_CORPUS_CSV)
print("✅ prepared_corpus loaded:", df.shape)

# 找文本列
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is None:
    raise ValueError(
        f"prepared_corpus.csv 找不到文本列。请确保存在以下任意一列：{TEXT_COL_CANDIDATES}\n"
        f"当前列：{list(df.columns)}"
    )

# 对齐检查：必须和 DTM 文档数一致
if len(df) != N_dtm:
    raise ValueError(
        f"❌ 行数不一致：prepared_corpus 有 {len(df)} 行，但 DTM gammas 有 {N_dtm} 行。\n"
        f"这说明 prepared_corpus 的顺序/范围可能和训练 DTM 时的语料不一致。\n"
        f"你需要用“训练 DTM 时的同一份、同一顺序”的原始表。"
    )

# quarter 列处理：优先用 prepared_corpus 自带 quarter
quarter_col = next((c for c in QUARTER_COL_CANDIDATES if c in df.columns), None)

if quarter_col is None:
    # 尝试用 time_slices.csv 补（可选）
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
# 3) 从 gammas 得到“真实 topic proportion”（θ -> quarter 平均）
# ======================
topic_cols = [f"Topic{i}" for i in range(K)]

# gammas -> theta（每篇文档归一化）
row_sum = gammas.sum(axis=1, keepdims=True).astype(float)
row_sum[row_sum == 0] = 1.0
theta = gammas / row_sum  # (N_docs, K)

theta_df = pd.DataFrame(theta, columns=topic_cols)
theta_df["quarter"] = df["quarter"].astype(str).values

# 每季度平均主题分布（更标准、可解释）
dtm_true_times = theta_df.groupby("quarter")[topic_cols].mean()
# 再做一次行归一化（防数值误差）
dtm_true_times = dtm_true_times.div(dtm_true_times.sum(axis=1), axis=0).fillna(0.0)
dtm_true_times = dtm_true_times.loc[sorted(dtm_true_times.index, key=qkey)]

# 导出 TRUE topic_times（加 _new）
out_true_csv = OUT_DIR / "EN_DTM_topic_times_TRUE_new.csv"
dtm_true_times.reset_index().rename(columns={"quarter": "Quarter"}).to_csv(
    out_true_csv, index=False, encoding="utf-8-sig"
)
print("✅ saved TRUE topic_times:", out_true_csv.resolve())

# 也给你一张全 topics 折线图（加 _new）
plt.figure(figsize=(12, 5))
for c in topic_cols:
    plt.plot(dtm_true_times.index, dtm_true_times[c].values, marker="o", label=c)
plt.xticks(rotation=35)
plt.ylabel("Proportion (TRUE, mean theta)")
plt.title("EN DTM Topic Proportion over Time (TRUE) _new")
plt.legend(ncol=3)
plt.tight_layout()
out_line_all = OUT_DIR / "EN_DTM_trend_line_TRUE_new.png"
plt.savefig(out_line_all, dpi=160)
plt.close()
print("✅ saved:", out_line_all.resolve())


# ======================
# 4) 给每篇文档一个 topic_id（用于柱子：硬分配计数）
#    建议用 theta.argmax（更干净）
# ======================
df["topic_id"] = theta.argmax(axis=1).astype(int)
df["topic"] = df["topic_id"].apply(lambda x: f"Topic{x}")

print("✅ topic prepared.")
print(df[["topic", "quarter"]].head())


# ======================
# 5) 加载 bert_lstm_model.pt 并做情感预测
# ======================
class BertSeqDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [str(x) for x in texts]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
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
        seq = out.last_hidden_state  # [B, L, H]

        lengths = attention_mask.sum(dim=1)  # [B]
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
        print(f"⚠️ 注意：你加载的 pt 里 model_type={model_type}，但你说要用 bert_lstm。")
    model_name = ckpt["model_name"]
    label2id = ckpt["label2id"]
    id2label = {i: c for c, i in label2id.items()}
    num_classes = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertLSTMClassifier(model_name, num_classes)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, tokenizer, id2label

sent_model, sent_tokenizer, id2label = load_sentiment_model(SENTIMENT_PT_PATH)

ds = BertSeqDataset(df[text_col].tolist(), sent_tokenizer, MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

pred_ids = []
with torch.no_grad():
    for batch in dl:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        logits = sent_model(input_ids, attn)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        pred_ids.extend(pred)

df["sent_pred"] = [id2label[i] for i in pred_ids]
df["sent_pred"] = df["sent_pred"].astype(str)

print("✅ sentiment predicted:", df["sent_pred"].value_counts().to_dict())


# ======================
# 6) 汇总表（加 _new）+ 保存明细
# ======================
tab_overall = (
    df.groupby(["topic", "sent_pred"])
      .size()
      .reset_index(name="count")
)
tab_q = (
    df.groupby(["quarter", "topic", "sent_pred"])
      .size()
      .reset_index(name="count")
)

out_overall_csv = OUT_DIR / "topic_sentiment_overall_new.csv"
out_byq_csv     = OUT_DIR / "topic_sentiment_by_quarter_new.csv"
out_detail_csv  = OUT_DIR / "prepared_with_topic_and_sentiment_new.csv"

tab_overall.to_csv(out_overall_csv, index=False, encoding="utf-8-sig")
tab_q.to_csv(out_byq_csv, index=False, encoding="utf-8-sig")
df.to_csv(out_detail_csv, index=False, encoding="utf-8-sig")

print("✅ saved tables:")
print(" -", out_overall_csv.resolve())
print(" -", out_byq_csv.resolve())
print(" -", out_detail_csv.resolve())


# ======================
# 7) 画 Overlay：每个 Topic 一张（柱：count；线：TRUE proportion）
# ======================
def plot_topic_overlay_one(
    tab_q: pd.DataFrame,
    dtm_true_times: pd.DataFrame,   # index=quarter, cols=Topic0..TopicK-1
    topic_name: str,                # e.g., "Topic0"
    out_png: Path,
    title_prefix: str = "Topic × Sentiment Over Time"
):
    # quarters 统一排序
    quarters = sorted(
        sorted(
            set(tab_q["quarter"].astype(str).tolist()) |
            set(dtm_true_times.index.astype(str).tolist())
        ),
        key=qkey
    )

    # 取这个 topic 的季度×sentiment counts
    sub = tab_q[tab_q["topic"] == topic_name].copy()
    if sub.empty:
        print(f"⚠️ 跳过：{topic_name} 在 tab_q 里没有数据")
        return

    pivot = sub.pivot_table(
        index="quarter",
        columns="sent_pred",
        values="count",
        aggfunc="sum",
        fill_value=0
    )
    pivot = pivot.reindex(quarters, fill_value=0)

    # 取这个 topic 的 TRUE intensity
    if topic_name not in dtm_true_times.columns:
        raise ValueError(f"dtm_true_times 里没有列 {topic_name}，现有列：{list(dtm_true_times.columns)}")

    line_y = dtm_true_times.reindex(quarters).fillna(0.0)[topic_name].values

    # 颜色：和你前面中文那套保持类似
    color_map = {
        "positive": "#2A9D8F",
        "negative": "#B56576",
        "pos": "#2A9D8F",
        "neg": "#B56576"
    }
    line_color = "#264653"

    fig, ax = plt.subplots(figsize=(12, 5))

    sentiments = pivot.columns.tolist()
    x = np.arange(len(quarters))
    width = 0.8 / max(1, len(sentiments))

    for j, s in enumerate(sentiments):
        ax.bar(
            x + (j - (len(sentiments)-1)/2) * width,
            pivot[s].values,
            width=width,
            label=f"{s} (count)",
            color=color_map.get(str(s).lower(), None),
            alpha=0.85
        )

    # 左轴
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=35, fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylabel("Count", fontsize=20)

    # 右轴
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

    # 标题
    en_topic_num = int(topic_name.replace("Topic", ""))
    ax.set_title(
        f"{title_prefix} — EN topic{en_topic_num}",
        fontsize=20,
        pad=10
    )

    # 合并 legend
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
        out_png = out_dir / f"topic_sentiment_over_time_{t}_new.png"
        plot_topic_overlay_one(tab_q, dtm_true_times, t, out_png)

plot_all_topics_overlay(tab_q, dtm_true_times, OUT_DIR)

print("🎉 DONE. 输出目录：", OUT_DIR.resolve())
print("关键输出：")
print(" - TRUE topic_times:", out_true_csv.resolve())
print(" - example overlay:", (OUT_DIR / "topic_sentiment_over_time_Topic0_new.png").resolve())
