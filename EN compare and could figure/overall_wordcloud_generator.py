# overall_wordcloud_generator.py
# -*- coding: utf-8 -*-
"""
整体词云图生成器（单张大图，不分 Topic）
- 支持两种数据来源：
  (A) dtm_aggregate: 用 DTM 主题词/权重聚合成“整体词云”（不展示 Topic）
  (B) corpus: 直接从语料 CSV 统计词频生成“整体词云”（更符合“主题建模之前”）
- 强制将 ai/Ai/a.i. 等统一为 'AI'，避免被拆成多个词导致不够大
"""

import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

try:
    from wordcloud import WordCloud, STOPWORDS
except ImportError:
    print("❌ 缺少 wordcloud 库，请安装：pip install wordcloud")
    raise


# ===================== 1) 配置区 =====================
BASE_DIR = Path(".")

# ------- 选择模式：推荐 corpus（更符合“topic modeling 之前”）-------
MODE = "dtm_aggregate"   # "dtm_aggregate" or "corpus"

# ------- DTM 输入（MODE=dtm_aggregate 时使用）-------
DTM_ALL_LONG = BASE_DIR / "TOPIC_OUT" / "dtm_topics_ALL_long.csv"
DTM_DIR = BASE_DIR / "TOPIC_OUT"   # fallback: dtm_topics_2023Q1.csv...

# ------- 语料输入（MODE=corpus 时使用）-------
# 英文：你现在的 prepared_corpus.csv 通常就够用
EN_CSV = BASE_DIR / "PREP" / "prepared_corpus.csv"
# 如果你有中文语料，也可以配置 CN_CSV（可选）
CN_CSV = None  # e.g., BASE_DIR / "PREP" / "weibo_ai_2023_2025_final_clean.csv"

# 文本列名：留 None 会自动猜（常见列：text/content/clean_text/...）
TEXT_COL_EN = None
TEXT_COL_CN = None

# 可选：只画某个季度（例如 "2024Q4"）；None = 全部季度
TARGET_QUARTER = None

# 可选：只用英文
ONLY_EN = True

# 词云最多用多少个词（太多会显得密）
TOP_WORDS = 300

# 输出目录
OUT_DIR = BASE_DIR / "WORDCLOUD_OUT"

# 是否显示标题（你如果想完全干净就 False）
SHOW_TITLE = False
TITLE = "Word Cloud"

# 词云图配置
WORDCLOUD_CONFIG = {
    'width': 1800,
    'height': 1100,
    'background_color': 'white',
    'colormap': 'viridis',
    'relative_scaling': 0.45,
    'min_font_size': 10,
    'max_font_size': 180,
    'prefer_horizontal': 0.75,
    'margin': 10,
}

# 图片质量
DPI = 300
FIGSIZE = (18, 11)

# 中文字体路径（需要中文词云时再填）
CHINESE_FONT = None  # e.g. "/System/Library/Fonts/PingFang.ttc"

# 英文额外停用词（你可继续加）
EXTRA_STOPWORDS_EN = {
    "rt", "amp", "https", "http", "www", "com", "co", "t", "u"
}


# ===================== 2) 工具函数 =====================
_QRE = re.compile(r"^(\d{4})Q([1-4])$")

def qkey(q: str):
    m = _QRE.match(str(q))
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)

def normalize_ai_token(s: str) -> str:
    """把 ai / Ai / a.i / A.I. 统一成 AI（只匹配整词，避免影响 chair 等）"""
    s = str(s)
    # 去掉两侧标点
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    # a.i / a.i. / ai / A.I. / Ai
    if re.fullmatch(r"a\.?i\.?", s2, flags=re.IGNORECASE):
        return "AI"
    return s2

def guess_text_col(df: pd.DataFrame):
    cands = ["text", "content", "clean_text", "processed_text", "post", "tweet", "body", "message"]
    for c in cands:
        if c in df.columns:
            return c
    # 兜底：选第一个 object 列
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[0] if obj_cols else None


# --------- DTM 相关（MODE=dtm_aggregate）---------
def _standardize_dtm_words(df: pd.DataFrame) -> pd.DataFrame:
    TOPIC_CANDS = ["topic", "topic_id", "topicId", "Topic"]
    WORD_CANDS = ["word", "term", "token"]
    RANK_CANDS = ["rank", "Rank"]
    WEIGHT_CANDS = ["weight", "Weight", "score", "prob"]
    QTR_CANDS = ["quarter", "Quarter", "time", "Time"]

    def pick(cands):
        return next((c for c in cands if c in df.columns), None)

    tcol = pick(TOPIC_CANDS)
    wcol = pick(WORD_CANDS)
    rcol = pick(RANK_CANDS)
    wtcol = pick(WEIGHT_CANDS)
    qcol = pick(QTR_CANDS)

    miss = []
    if tcol is None: miss.append("topic/topic_id")
    if wcol is None: miss.append("word")
    if rcol is None: miss.append("rank")
    if miss:
        raise AssertionError(f"dtm_topics 文件缺少必要列：{miss}；当前列={list(df.columns)}")

    out = df.copy().rename(columns={tcol: "topic", wcol: "word", rcol: "rank"})
    if wtcol is None:
        out["weight"] = 1.0
    else:
        out = out.rename(columns={wtcol: "weight"})
    if qcol is not None:
        out = out.rename(columns={qcol: "quarter"})

    out["topic"] = pd.to_numeric(out["topic"], errors="coerce").astype("Int64")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0).astype(float)
    out["word"] = out["word"].astype(str).map(normalize_ai_token)

    out = out[out["topic"].notna() & out["rank"].notna()].copy()
    out["topic"] = out["topic"].astype(int)
    out["rank"] = out["rank"].astype(int)

    cols = ["topic", "rank", "word", "weight"] + (["quarter"] if "quarter" in out.columns else [])
    return out[cols].copy()

def load_dtm_words():
    if DTM_ALL_LONG.exists():
        df = pd.read_csv(DTM_ALL_LONG, encoding="utf-8-sig")
        df = _standardize_dtm_words(df)
        if TARGET_QUARTER is not None and "quarter" in df.columns:
            df = df[df["quarter"].astype(str) == str(TARGET_QUARTER)].copy()
        return df

    parts = []
    for p in DTM_DIR.glob("dtm_topics_*.csv"):
        m = re.match(r"dtm_topics_(\d{4}Q[1-4])\.csv$", p.name)
        if not m:
            continue
        q = m.group(1)
        if TARGET_QUARTER is not None and q != str(TARGET_QUARTER):
            continue
        tmp = pd.read_csv(p, encoding="utf-8-sig")
        tmp = _standardize_dtm_words(tmp)
        tmp["quarter"] = q
        parts.append(tmp)

    if not parts:
        raise RuntimeError("找不到 dtm_topics_ALL_long.csv，也没找到 dtm_topics_<Q>.csv")

    return pd.concat(parts, ignore_index=True)

def get_global_word_weights_from_dtm(dtm_df: pd.DataFrame, topn: int):
    """
    把所有 topic 的词聚合成整体权重：
    1) 若有 quarter：先对 (topic, word) 在季度上取 mean
    2) 再对 word 在 topic 上求 sum -> 体现“跨主题总体重要性”
    """
    df = dtm_df.copy()
    if "quarter" in df.columns:
        df1 = (df.groupby(["topic", "word"], as_index=False)["weight"]
               .mean())
    else:
        df1 = df[["topic", "word", "weight"]].copy()

    df2 = (df1.groupby(["word"], as_index=False)["weight"]
           .sum()
           .sort_values("weight", ascending=False))

    df2 = df2.head(topn)
    return dict(zip(df2["word"], df2["weight"]))


# --------- 语料相关（MODE=corpus）---------
def tokenize_en(text: str):
    # 只保留“像单词”的东西（可按你清洗结果再调整）
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+\-]*", str(text))
    out = []
    for t in tokens:
        t = normalize_ai_token(t)
        if not t:
            continue
        low = t.lower()
        if low in STOPWORDS or low in EXTRA_STOPWORDS_EN:
            continue
        out.append("AI" if low == "ai" else t)  # 双保险
    return out

def build_word_weights_from_corpus(csv_path: Path, text_col: str | None, topn: int, lang: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col is None:
        text_col = guess_text_col(df)
    if text_col is None or text_col not in df.columns:
        raise RuntimeError(f"找不到文本列。请手动设置 TEXT_COL_{lang.upper()}，当前列={list(df.columns)}")

    counter = Counter()
    for s in df[text_col].astype(str).fillna(""):
        if lang == "en":
            counter.update(tokenize_en(s))
        else:
            # 中文如果你后面要用，我建议用 jieba 分词；这里先留最简兜底（按空格切）
            toks = [normalize_ai_token(x) for x in str(s).split() if x.strip()]
            counter.update(toks)

    most = counter.most_common(topn)
    return dict(most), text_col


# --------- 绘图 ---------
def generate_wordcloud(word_weights: dict, out_path: Path, title: str = ""):
    if not word_weights:
        print("⚠️ 无有效词，跳过")
        return

    wc_config = WORDCLOUD_CONFIG.copy()
    if CHINESE_FONT and Path(CHINESE_FONT).exists():
        wc_config['font_path'] = str(CHINESE_FONT)

    wc = WordCloud(**wc_config).generate_from_frequencies(word_weights)

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if SHOW_TITLE and title:
        plt.title(title, fontsize=22, weight='bold', pad=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=DPI)
    plt.close()
    print(f"✅ saved: {out_path.resolve()}")


# ===================== 3) 主流程 =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{TARGET_QUARTER}" if TARGET_QUARTER else ""
    if ONLY_EN:
        out_png = OUT_DIR / f"wordcloud_overall_en{suffix}.png"
    else:
        out_png = OUT_DIR / f"wordcloud_overall{suffix}.png"

    if MODE == "dtm_aggregate":
        print("📖 Loading DTM topic words...")
        dtm_df = load_dtm_words()
        print("🧮 Aggregating across topics -> overall word weights...")
        word_weights = get_global_word_weights_from_dtm(dtm_df, topn=TOP_WORDS)

        # 再做一次 AI 统一（双保险）
        fixed = {}
        for w, wt in word_weights.items():
            ww = normalize_ai_token(w)
            fixed[ww] = fixed.get(ww, 0.0) + float(wt)
        word_weights = fixed

        title = TITLE + (f" ({TARGET_QUARTER})" if TARGET_QUARTER else "")
        generate_wordcloud(word_weights, out_path=out_png, title=title)

    elif MODE == "corpus":
        if ONLY_EN:
            if not EN_CSV or not Path(EN_CSV).exists():
                raise RuntimeError(f"找不到 EN_CSV：{EN_CSV}")
            print(f"📖 Loading corpus: {EN_CSV}")
            word_weights, used_col = build_word_weights_from_corpus(EN_CSV, TEXT_COL_EN, TOP_WORDS, lang="en")
            print(f"✅ Using text column: {used_col}")
            title = TITLE + (f" ({TARGET_QUARTER})" if TARGET_QUARTER else "")
            generate_wordcloud(word_weights, out_path=out_png, title=title)
        else:
            if not CN_CSV or not Path(CN_CSV).exists():
                raise RuntimeError(f"找不到 CN_CSV：{CN_CSV}")
            word_weights, used_col = build_word_weights_from_corpus(CN_CSV, TEXT_COL_CN, TOP_WORDS, lang="cn")
            print(f"✅ Using text column: {used_col}")
            title = TITLE + (f" ({TARGET_QUARTER})" if TARGET_QUARTER else "")
            generate_wordcloud(word_weights, out_path=out_png, title=title)

    else:
        raise ValueError("MODE 必须是 'dtm_aggregate' 或 'corpus'")

    print(f"\n🎉 DONE -> {OUT_DIR.resolve()}")
    print(f"   Output: {out_png.name}")


if __name__ == "__main__":
    main()