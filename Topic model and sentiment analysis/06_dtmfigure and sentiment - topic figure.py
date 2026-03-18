# -*- coding: utf-8 -*-dtm图像和情感分析图像
"""
CN: TRUE DTM topic_times + Topic×Sentiment overlay (NEW)
- Model: my_dtm_model.gensim (gensim.models.ldaseqmodel.LdaSeqModel)
- TRUE topic_times: from gammas + time_slice (doc-normalize -> slice-mean -> normalize)
- Quarters: derived from prepared_corpus.csv by time_slice blocks (mode quarter per slice) => strict alignment
- Sentiment: reuse doc_topic_sentiment.csv (no inference). If row-count matches, align by index.
Outputs (all with _new):
- cn_topic_sentiment_out_new/CN_DTM_topic_times_TRUE_new.csv
- cn_topic_sentiment_out_new/CN_DTM_trend_line_TRUE_new.png
- cn_topic_sentiment_out_new/CN_DTM_trend_stacked_TRUE_new.png
- cn_topic_sentiment_out_new/topic_sentiment_by_quarter_new.csv
- cn_topic_sentiment_out_new/00_xxx_overlay_new.png ... 05_xxx_overlay_new.png
"""

import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 0)路径
# ======================
DTM_MODEL_PATH      = Path("my_dtm_model.gensim")
PREPARED_CORPUS_CSV = Path("weibo_ai_dtm_train_subset_47137.csv")
DOC_SENT_CSV        = Path("doc_topic_sentiment.csv")     

OUT_DIR = Path("cn_topic_sentiment_out_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 主题命名（你论文用的）
TOPIC_TO_FINAL = {
    0: "海外大厂动向",
    1: "技术应用进展",
    2: "人机伦理思考",
    3: "产业发展政策",
    4: "A股市场风向",
    5: "教育科研推广",
}

# 颜色：与英文一致
C_POS  = "#2A9D8F"
C_NEG  = "#B56576"
C_LINE = "#264653"

# 字体（Mac）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# prepared_corpus 可能的列名
TEXT_COL_CANDIDATES = ["text_raw", "text_norm", "text", "content", "full_text"]
QUARTER_COL_CANDIDATES = ["quarter", "Quarter", "time", "Time", "slice", "Slice"]

# doc_topic_sentiment 可能的情感列名
SENT_COL_CANDIDATES = ["sent_pred", "sent_label", "sentiment", "pred", "label"]

# ======================
# 1) 加载 LdaSeqModel
# ======================
def load_ldaseq_model(path: Path):
    path = Path(path)
    assert path.exists(), f"找不到 DTM 模型：{path.resolve()}"
    from gensim.models.ldaseqmodel import LdaSeqModel
    return LdaSeqModel.load(str(path))

dtm = load_ldaseq_model(DTM_MODEL_PATH)
K = int(dtm.num_topics)
T = int(dtm.num_time_slices)
time_slice = list(dtm.time_slice)
gammas = np.asarray(dtm.gammas, dtype=float)  # (D, K)

print("✅ DTM loaded:", type(dtm))
print("   K =", K, "| T =", T)
print("   time slices =", len(time_slice), " total docs =", sum(time_slice))
print("   gammas shape =", gammas.shape)

assert K == 6, f"你现在K={K}，不是6的话请检查模型"
assert len(time_slice) == T
assert sum(time_slice) == gammas.shape[0]

N = gammas.shape[0]

# ======================
# 2) 读取 prepared_corpus，并生成严格对齐的 Quarter 序列
#    关键：按 time_slice 切块，每块取 quarter 众数 => quarters_use 与模型时间片严格一致
# ======================
assert PREPARED_CORPUS_CSV.exists(), f"找不到：{PREPARED_CORPUS_CSV.resolve()}"
df = pd.read_csv(PREPARED_CORPUS_CSV)
print("✅ prepared_corpus loaded:", df.shape)

if len(df) != N:
    raise ValueError(
        f"❌ 行数不一致：prepared_corpus={len(df)} vs DTM docs={N}\n"
        f"必须使用“训练DTM时同一份、同一顺序”的 prepared_corpus.csv"
    )

# quarter 列：优先用 prepared_corpus 自带的 quarter
quarter_col = next((c for c in QUARTER_COL_CANDIDATES if c in df.columns), None)

if quarter_col is None:
    # 如果没有 quarter，用 time_slices.csv 映射 time->quarter
    if TIME_SLICES_CSV.exists():
        ts = pd.read_csv(TIME_SLICES_CSV)
        tcol = next((c for c in ["time", "Time", "t"] if c in ts.columns), None)
        qcol = next((c for c in ["quarter", "Quarter"] if c in ts.columns), None)
        if tcol and qcol:
            mapping = dict(zip(ts[tcol].tolist(), ts[qcol].astype(str).tolist()))
            time_idx = np.concatenate([np.full(n, i) for i, n in enumerate(time_slice)])
            df["quarter"] = [mapping.get(int(t), f"time{int(t)}") for t in time_idx]
            quarter_col = "quarter"
        else:
            time_idx = np.concatenate([np.full(n, i) for i, n in enumerate(time_slice)])
            df["quarter"] = [f"time{int(t)}" for t in time_idx]
            quarter_col = "quarter"
    else:
        time_idx = np.concatenate([np.full(n, i) for i, n in enumerate(time_slice)])
        df["quarter"] = [f"time{int(t)}" for t in time_idx]
        quarter_col = "quarter"

# 清洗 quarter 格式
df["_quarter_clean"] = (
    df[quarter_col].astype(str)
      .str.upper()
      .str.replace(r"\s+", "", regex=True)
)

# 按 time_slice 切块，取众数 quarter => 作为该时间片的 quarter 标签
quarters_use = []
s = 0
for i, n in enumerate(time_slice):
    block_q = df["_quarter_clean"].iloc[s:s+n]
    if len(block_q) == 0:
        quarters_use.append(f"time{i}")
    else:
        # 众数（出现最多的quarter）
        quarters_use.append(block_q.value_counts().idxmax())
    s += n

print("✅ quarters (aligned to time_slice):", quarters_use)

# ======================
# 3) 计算 TRUE topic_times（每时间片 topic 占比）
# ======================
topic_times = []
s = 0
for n in time_slice:
    block = gammas[s:s+n]  # (n, K)

    # 每篇文档归一化 => theta
    rs = block.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    theta = block / rs

    # 时间片内平均，再归一化（每片和=1）
    m = theta.mean(axis=0)
    m = m / (m.sum() if m.sum() != 0 else 1.0)

    topic_times.append(m)
    s += n

topic_times = np.vstack(topic_times)  # (T,K)
cols = [f"Topic{i}" for i in range(K)]
df_true = pd.DataFrame(topic_times, columns=cols)
df_true.insert(0, "Quarter", quarters_use)

# 保存 TRUE topic_times
TRUE_CSV = OUT_DIR / "CN_DTM_topic_times_TRUE_new.csv"
df_true.to_csv(TRUE_CSV, index=False, encoding="utf-8-sig")
print("✅ saved TRUE topic_times:", TRUE_CSV.resolve())

# 画 TRUE 折线图（可选，但建议保留）
fig, ax = plt.subplots(figsize=(12, 5))

for c in cols:
    ax.plot(df_true["Quarter"], df_true[c], marker="o", linewidth=2.2, markersize=8, label=c)

ax.set_ylabel("Topic Intensity", fontsize=20, fontweight="bold")
ax.set_title("CN DTM Topic Intensity over Time", fontsize=20, fontweight="bold")

ax.tick_params(axis='x', labelsize=20, rotation=35, length=6, width=1.2, direction='out')
ax.tick_params(axis='y', labelsize=15, length=6, width=1.2, direction='out')

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.legend(ncol=3, fontsize=11)
fig.tight_layout()
LINE_PNG = OUT_DIR / "CN_DTM_trend_line_TRUE_new.png"
plt.savefig(LINE_PNG, dpi=160)
plt.close()
print("✅ saved:", LINE_PNG.resolve())

# 画 TRUE 堆叠面积图
plt.figure(figsize=(12,5))
plt.stackplot(df_true["Quarter"], *[df_true[c].to_numpy(float) for c in cols], labels=cols)
plt.xticks(rotation=35)
plt.ylabel("Proportion (TRUE)")
plt.title("CN DTM Topic Proportion over Time (Stacked, TRUE)_new")
plt.legend(ncol=3, loc="upper center")
plt.tight_layout()
STACK_PNG = OUT_DIR / "CN_DTM_trend_stacked_TRUE_new.png"
plt.savefig(STACK_PNG, dpi=160)
plt.close()
print("✅ saved:", STACK_PNG.resolve())

# ======================
# 4) 生成“每条文档 topic_id / quarter”（用于计数）
# ======================
df_doc = pd.DataFrame({
    "quarter": df["_quarter_clean"].tolist(),
})
df_doc["topic_id"] = gammas.argmax(axis=1).astype(int)   # doc 的主主题
df_doc["topic"] = df_doc["topic_id"].apply(lambda x: f"Topic{x}")

# ======================
# 5) 读取情感预测（复用，不再推理）
#    优先：如果 DOC_SENT_CSV 行数==N，则按index直接拼接 sent_pred
#    否则：尝试从里面找 quarter/topic/sent_pred 直接聚合（退化方案）
# ======================
def normalize_sent_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in ("1", "pos", "positive", "true", "yes"):
        return "positive"
    if s in ("0", "neg", "negative", "false", "no"):
        return "negative"
    # 已经是 positive/negative 就原样
    if s in ("positive", "negative"):
        return s
    # 其他就原样返回（但图里可能会多出类别）
    return s

sent_ok = False
if DOC_SENT_CSV.exists():
    ds = pd.read_csv(DOC_SENT_CSV, encoding="utf-8-sig")
    sent_col = next((c for c in SENT_COL_CANDIDATES if c in ds.columns), None)

    if sent_col is not None and len(ds) == N:
        # ✅最稳：按行对齐
        df_doc["sent_pred"] = ds[sent_col].map(normalize_sent_label).tolist()
        sent_ok = True
        print("✅ sentiment loaded by row-alignment from:", DOC_SENT_CSV.resolve())
    else:
        print("⚠️ doc_topic_sentiment.csv 不能行对齐（可能行数不同或缺少情感列），将尝试退化聚合。")

if not sent_ok:
    raise ValueError(
        "❌ 你这个脚本要求复用已预测的情感结果，但 doc_topic_sentiment.csv 无法按行对齐。\n"
        "请确保 doc_topic_sentiment.csv：\n"
        "1) 行数与 prepared_corpus.csv / DTM docs 完全一致；\n"
        "2) 包含情感列（sent_pred / sent_label / sentiment / pred / label 任意一个）。"
    )

print("✅ sentiment distribution:", df_doc["sent_pred"].value_counts().to_dict())

# ======================
# 6) 聚合：quarter × topic × sent => pos/neg count
# ======================
# 只保留 positive/negative（如果你的数据里有别的标签，这里会过滤掉）
df_doc = df_doc[df_doc["sent_pred"].isin(["positive", "negative"])].copy()

tab_q = (
    df_doc.groupby(["quarter", "topic", "sent_pred"])
          .size()
          .reset_index(name="count")
)

TAB_Q_CSV = OUT_DIR / "topic_sentiment_by_quarter_new.csv"
tab_q.to_csv(TAB_Q_CSV, index=False, encoding="utf-8-sig")
print("✅ saved:", TAB_Q_CSV.resolve())

# ======================
# 7) 画每个主题：柱(情感count) + 线(TRUE topic proportion)
# ======================
quarters_plot = df_true["Quarter"].astype(str).tolist()

# 把 df_true 的 Topic0..5 转成 dict 方便取线
true_line = {f"Topic{i}": df_true[f"Topic{i}"].to_numpy(float) for i in range(K)}

def overlay_png(k: int, label: str) -> Path:
    safe = str(label).replace("/", "_").replace("\\", "_").replace(" ", "")
    return OUT_DIR / f"{k:02d}_{safe}_overlay_new.png"

for k in range(K):
    topic_name = TOPIC_TO_FINAL.get(k, f"Topic{k}")
    topic_key = f"Topic{k}"

    # 取该 topic 在各季度的 TRUE 占比（折线）
    y_line = true_line[topic_key]

    # 取该 topic 各季度的 pos/neg count（柱子）
    sub = tab_q[tab_q["topic"] == topic_key].copy()
    pv = sub.pivot(index="quarter", columns="sent_pred", values="count").fillna(0)
    pv = pv.reindex(quarters_plot).fillna(0)

    pos = pv["positive"].to_numpy(float) if "positive" in pv.columns else np.zeros(len(quarters_plot))
    neg = pv["negative"].to_numpy(float) if "negative" in pv.columns else np.zeros(len(quarters_plot))

    x = np.arange(len(quarters_plot))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 柱：count
    ax1.bar(x - w/2, pos, width=w, label="Positive (count)", color=C_POS, alpha=0.85)
    ax1.bar(x + w/2, neg, width=w, label="Negative (count)", color=C_NEG, alpha=0.85)

    ax1.set_ylabel("Count", fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarters_plot, rotation=35, fontsize=20)
    ax1.tick_params(axis='y', labelsize=15)

    # 线：TRUE proportion
    ax2 = ax1.twinx()
    ax2.plot(x, y_line, marker="o", linewidth=2.5, label="Topic intensity", color=C_LINE)
    ax2.set_ylabel("Topic intensity", fontsize=20)
    ax2.tick_params(axis='y', labelsize=15)

    ax1.set_title(f"Topic × Sentiment Over Time — CN topic{k}", fontsize=20)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fontsize=11)

    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    out_png = overlay_png(k, topic_name)
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"[debug] {topic_name} TRUE proportions:", np.round(y_line, 6).tolist())
    print("✅ saved:", out_png.resolve())

print("🎉 DONE. 输出目录：", OUT_DIR.resolve())
