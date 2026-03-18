# -*- coding: utf-8 -*-词云图
"""
CN Overall WordCloud (ONE big image) from DTM txt outputs
- Read: 10.14工作_DTM_topics_time=2023Q1.txt ... 2025Q3.txt
- Aggregate across ALL quarters (2023Q1-2025Q3) AND ALL topics (K=6)
- Output ONLY 1 image: wordcloud_global_CN.png
- Normalize ai/Ai/a.i./A.I. -> AI
"""

import re
import ast
from pathlib import Path
from collections import defaultdict

import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ===================== 1) CONFIG =====================
BASE_DIR = Path(".")         # txt 文件所在目录（默认当前目录）
OUT_DIR  = Path("OUT_WORDCLOUD_CN")

TXT_PREFIX = "DTM_topics_time="

QUARTERS = [
    "2023Q1","2023Q2","2023Q3","2023Q4",
    "2024Q1","2024Q2","2024Q3","2024Q4",
    "2025Q1","2025Q2","2025Q3",
]

K = 6  # topic count

# 词云图更清晰一点
WC_WIDTH  = 1800
WC_HEIGHT = 1100
WC_MAX_WORDS = 180
WC_BACKGROUND = "white"

# 可选：额外过滤一些泛词（你可以按需增删）
EXTRA_STOPWORDS = {
    "真的","感觉","知道","觉得","可能","时候","需要","问题","内容","用户",
    "一个","我们","你们","他们","它们","以及","因为","所以"
}

# 聚合方式（整体词云）
# - "mean": 对每个词在出现过的“(季度×topic)”位置取平均权重（推荐，稳定）
# - "sum" : 直接把所有出现位置权重加总（会更偏向出现次数多的词）
AGG_MODE = "mean"   # 改成 "sum" 也行

# 标题：你说“不要 topic 单列”，且这部分在 topic modeling 之前，默认不加标题
SHOW_TITLE = False
TITLE_TEXT = "Overall Word Cloud (CN)"   # SHOW_TITLE=True 时才会用


# ===================== 2) FONT (fix OSError) =====================
def pick_cn_font():
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/STSongti-SC-Regular.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",

        # Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",

        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return None

FONT_PATH = pick_cn_font()
if FONT_PATH is None:
    raise OSError(
        "❌ 找不到可用中文字体文件（wordcloud 需要 font_path）。\n"
        "请手动把你机器上的中文字体 .ttf/.ttc 路径填到 FONT_PATH。\n"
        "macOS 常用：/System/Library/Fonts/PingFang.ttc\n"
        "Windows 常用：C:/Windows/Fonts/msyh.ttc"
    )
print("✅ Using font:", FONT_PATH)


# ===================== 3) NORMALIZER =====================
def normalize_ai_token(w: str) -> str:
    """
    把 ai / Ai / a.i / A.I. 统一成 AI（仅匹配“整词”形式）
    """
    s = str(w).strip()
    # 去掉两侧标点
    s2 = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    # a.i / a.i. / ai / A.I. / Ai
    if re.fullmatch(r"a\.?i\.?", s2, flags=re.IGNORECASE):
        return "AI"
    return s2


# ===================== 4) PARSER =====================
_Q_RE = re.compile(r"time\s*=\s*(\d{4}Q[1-4])", re.I)

def parse_one_txt(path: Path, k: int = 6):
    """
    return:
      quarter(str), topics(list of list[(word, weight)])
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    # quarter
    quarter = None
    if lines:
        m = _Q_RE.search(lines[0])
        if m:
            quarter = m.group(1)
    if not quarter:
        mm = re.search(r"time=(\d{4}Q[1-4])", path.name)
        quarter = mm.group(1) if mm else "UNKNOWN"

    # parse next k topic lines that start with '['
    topic_lines = []
    for ln in lines[1:]:
        if ln.startswith("["):
            topic_lines.append(ln)
        if len(topic_lines) >= k:
            break
    if len(topic_lines) < k:
        raise ValueError(f"{path.name} topic 行不足 {k}（只找到 {len(topic_lines)}）")

    topics = []
    for i, ln in enumerate(topic_lines):
        obj = ast.literal_eval(ln)  # list of tuples
        pairs = []
        for it in obj:
            if not isinstance(it, (tuple, list)) or len(it) < 2:
                continue
            w = normalize_ai_token(it[0])
            if (not w) or (w in EXTRA_STOPWORDS):
                continue
            try:
                wt = float(it[1])
            except Exception:
                continue
            pairs.append((w, wt))
        topics.append(pairs)

    return quarter, topics


# ===================== 5) DRAW =====================
def draw_wordcloud(freq: dict, out_png: Path):
    if not freq:
        print("⚠️ skip empty:", out_png.name)
        return

    wc = WordCloud(
        font_path=FONT_PATH,
        width=WC_WIDTH,
        height=WC_HEIGHT,
        background_color=WC_BACKGROUND,
        max_words=WC_MAX_WORDS,
        prefer_horizontal=0.95,
        collocations=False
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(WC_WIDTH / 160, WC_HEIGHT / 160), dpi=220)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if SHOW_TITLE:
        plt.title(TITLE_TEXT, fontsize=16)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ saved:", out_png.resolve())


# ===================== 6) MAIN =====================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # locate txt files
    txt_files = []
    for q in QUARTERS:
        p = BASE_DIR / f"{TXT_PREFIX}{q}.txt"
        if p.exists():
            txt_files.append(p)

    print(f"✅ found txt files: {len(txt_files)}")
    if not txt_files:
        raise FileNotFoundError(
            f"找不到任何 txt：{TXT_PREFIX}2023Q1.txt 这种命名。\n"
            "请确认 txt 是否在当前目录。"
        )

    # read all quarters -> topics
    quarter_to_topics = {}
    quarters_ok = []
    for p in txt_files:
        q, topics = parse_one_txt(p, k=K)
        quarter_to_topics[q] = topics
        quarters_ok.append(q)

    quarters_ok = sorted(quarters_ok, key=lambda x: (int(x[:4]), int(x[-1])))
    print("✅ quarters:", quarters_ok)

    # ===== 生成“整体”词云：聚合 ALL topics + ALL quarters =====
    agg_list = defaultdict(list)  # word -> list[weights]

    for q in quarters_ok:
        topics = quarter_to_topics[q]

        for k in range(K):
            pairs = topics[k]

            # 每个 (q,k) 内同词去重：取最大权重，避免重复
            d = {}
            for w, wt in pairs:
                if (not w) or (w in EXTRA_STOPWORDS):
                    continue
                w = normalize_ai_token(w)
                d[w] = max(d.get(w, 0.0), float(wt))

            for w, wt in d.items():
                agg_list[w].append(wt)

    if AGG_MODE.lower() == "sum":
        freq_global = {w: float(np.sum(wts)) for w, wts in agg_list.items()}
    else:
        freq_global = {w: float(np.mean(wts)) for w, wts in agg_list.items()}

    # 再把 AI 做一次双保险合并（如果同时出现 ai/AI）
    merged = defaultdict(float)
    for w, wt in freq_global.items():
        ww = normalize_ai_token(w)
        merged[ww] += float(wt)
    freq_global = dict(merged)

    out_png = OUT_DIR / "wordcloud_global_CN.png"
    draw_wordcloud(freq_global, out_png)

    print("🎉 DONE ->", OUT_DIR.resolve())


if __name__ == "__main__":
    main()