import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import seaborn as sns

# ✅ 只为 matched 增补：匈牙利算法
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

# ===================== 1. 配置区域 =====================
BASE_DIR = Path(".")
DTM_DIR  = BASE_DIR /  "TOPIC_OUT"

# ✅ 你的全局 BERTopic 文件（相对路径）
BERT_GLOBAL_PATH = BASE_DIR / "EN_BERTopic_Out" / "bert_topics_en.csv"

# ✅ 向量模型：二选一
# 1) 如果你本地有模型文件夹，用本地路径：
# EMBED_MODEL_PATH = BASE_DIR / "models" / "bge-large-en-v1.5"
# 2) 如果你没有本地模型且可联网，直接写模型名：
EMBED_MODEL_PATH = "BAAI/bge-large-en-v1.5"

QUARTERS = [
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    "2025Q1", "2025Q2", "2025Q3",
]

TOPN = 30

DOMAIN_STOPWORDS = {
    "ai", "artificial", "intelligence", "learning", "data",
    "model", "technology", "human", "use", "new", "time", "based",
    "systems", "development", "potential", "future", "world"
}

SAVE_FIG_NAME = "heatmap_dtm_vs_bert.png"

# ✅ matched 图另存一个
SAVE_FIG_NAME_MATCHED = "heatmap_dtm_vs_bert_matched.png"

# ===================== 2. 辅助函数 =====================

def process_words_to_text(df, topic_id, topn=TOPN):
    words = df[df["topic"] == topic_id].sort_values("rank").head(topn)["word"].tolist()
    filtered_words = [str(w) for w in words if str(w).lower() not in DOMAIN_STOPWORDS and str(w) != 'nan']
    if not filtered_words:
        filtered_words = [str(w) for w in words if str(w) != 'nan']
    return " ".join(filtered_words)

def load_global_bert():
    path = Path(BERT_GLOBAL_PATH)
    if not path.exists():
        print(f"❌ 找不到全局 BERTopic 文件: {path}")
        return None, None
    df = pd.read_csv(path)
    topics = sorted(df["topic"].unique().tolist())
    topic_texts = []
    for t in topics:
        text = process_words_to_text(df, t)
        topic_texts.append(text)
    print(f"✅ 成功加载全局 BERTopic，共 {len(topics)} 个主题。")
    return topics, topic_texts

def load_dtm_quarter(quarter):
    csv_path = DTM_DIR / f"dtm_topics_{quarter}.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    topics = sorted(df["topic"].unique().tolist())
    topic_texts = []
    for t in topics:
        text = process_words_to_text(df, t)
        topic_texts.append(text)
    return topics, topic_texts

# ===================== 3. 主流程 =====================

def main():
    print(f"正在加载模型: {EMBED_MODEL_PATH} ...")
    try:
        model = SentenceTransformer(str(EMBED_MODEL_PATH))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 1. 准备全局向量
    bert_ids, bert_texts = load_global_bert()
    if bert_ids is None:
        return
    bert_emb_global = model.encode(bert_texts, normalize_embeddings=False)

    # 2. 准备动态向量
    all_dtm_embs = []
    quarter_records = []

    print("正在处理各季度 DTM 数据...")
    for q in QUARTERS:
        dtm_ids, dtm_texts = load_dtm_quarter(q)
        if dtm_ids is not None:
            dtm_emb = model.encode(dtm_texts, normalize_embeddings=False)
            all_dtm_embs.append(dtm_emb)
            quarter_records.append({
                "q": q, "d_ids": dtm_ids, "d_emb": dtm_emb
            })

    if not all_dtm_embs:
        print("❌ 未找到任何有效的 DTM 数据")
        return

    # 3. 计算中心向量 (去共性)
    concat_list = [bert_emb_global] + all_dtm_embs
    global_mean_vec = np.mean(np.vstack(concat_list), axis=0)

    # 4. 计算相似度
    final_sims = []
    dtm_ids_check = None

    bert_emb_centered = bert_emb_global - global_mean_vec

    for item in quarter_records:
        dtm_emb_centered = item["d_emb"] - global_mean_vec
        sim = cosine_similarity(dtm_emb_centered, bert_emb_centered)
        final_sims.append(sim)
        if dtm_ids_check is None:
            dtm_ids_check = item["d_ids"]

    if not final_sims:
        print("❌ 没有生成相似度矩阵，无法绘图。")
        return

    # 取平均
    mean_sim = np.stack(final_sims, axis=0).mean(axis=0)

    # 归一化到 0-1
    min_val = mean_sim.min()
    max_val = mean_sim.max()
    mean_sim_norm = (mean_sim - min_val) / (max_val - min_val + 1e-12)

    print("✅ 计算完成。已归一化到 0-1 区间。")

    # ====== 原图（不匹配）======
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        mean_sim_norm,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 20},
        cmap="YlGnBu",
        vmin=0.0, vmax=1.0,
        xticklabels=[f"BERTopic {t}" for t in bert_ids],
        yticklabels=[f"DTM {t}" for t in dtm_ids_check],
        square=True,
        cbar_kws={"label": "Normalized Similarity Score (0-1)"}
    )

    plt.title("Topic Correspondence: Dynamic DTM vs Quarterly BERTopic\n(0=Least Similar, 1=Most Similar)", fontsize=14)
    plt.xlabel("BERTopic Topics", fontsize=12)
    plt.ylabel("DTM Topics", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    print(f"正在保存图片到: {SAVE_FIG_NAME}")
    plt.savefig(SAVE_FIG_NAME, dpi=300, bbox_inches="tight")
    print("✅ 图片保存成功！")
    plt.show()

    # ====== ✅ 只新增：matched 图 ======
    if linear_sum_assignment is None:
        print("⚠️ scipy 不可用，无法进行 matched（匈牙利算法）重排。")
        return

    r_ind, c_ind = linear_sum_assignment(-mean_sim_norm)  # 最大化相似度
    matched = mean_sim_norm[r_ind][:, c_ind]

    dtm_labels_m = [f"DTM {dtm_ids_check[i]}" for i in r_ind]
    bert_labels_m = [f"BERTopic {bert_ids[j]}" for j in c_ind]

    plt.figure(figsize=(10, 10))
    ax2 = sns.heatmap(
        matched,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 20},
        cmap="YlGnBu",
        vmin=0.0, vmax=1.0,
        xticklabels=bert_labels_m,
        yticklabels=dtm_labels_m,
        square=True,
        cbar_kws={"label": "Normalized Similarity Score (0-1)"}
    )
    cbar = ax2.collections[0].colorbar
    cbar.set_label("Normalized Similarity Score (0-1)", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.title("Topic Correspondence: Dynamic DTM vs Quarterly BERTopic \n(0=Least Similar, 1=Most Similar)", fontsize=20)
    plt.xlabel("BERTopic Topics", fontsize=20)
    plt.ylabel("DTM Topics", fontsize=20)
    plt.xticks(rotation=35, ha="right", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.tight_layout()

    print(f"正在保存 matched 图片到: {SAVE_FIG_NAME_MATCHED}")
    plt.savefig(SAVE_FIG_NAME_MATCHED, dpi=300, bbox_inches="tight")
    print("✅ matched 图片保存成功！")
    plt.show()

if __name__ == "__main__":
    main()
