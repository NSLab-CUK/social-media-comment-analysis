# === Step 1: 抽样一批英文样本，生成人工标注模板 ===
import os, random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED)

IN_CSV   = Path("PREP/prepared_corpus.csv")
OUT_DIR  = Path("Teacher_Cls"); OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLE = 800   # 建议 600~1000，看你精力，800 比较稳

assert IN_CSV.exists(), f"找不到 {IN_CSV}"
df = pd.read_csv(IN_CSV)

# 选文本列：优先 text_raw
text_col = "text_raw" if "text_raw" in df.columns else (
           "text_norm" if "text_norm" in df.columns else None)
assert text_col is not None, "需要 text_raw 或 text_norm 列"

# 如果有 lang 列，可以只保留英文
if "lang" in df.columns:
    df_lang = df[df["lang"].astype(str).str.lower().str.startswith("en")].copy()
    if not df_lang.empty:
        df = df_lang

print("总样本数：", len(df))

# 抽样
df_sample = df.sample(min(N_SAMPLE, len(df)), random_state=SEED).copy()

keep_cols = []
if "doc_id" in df_sample.columns:
    keep_cols.append("doc_id")
keep_cols.append(text_col)
if "quarter" in df_sample.columns:
    keep_cols.append("quarter")

df_sample = df_sample[keep_cols].copy()
df_sample["label"] = ""  # 留空等你填：positive / negative

out_path = OUT_DIR / "en_gold_template.csv"
df_sample.to_csv(out_path, index=False, encoding="utf-8-sig")
print("✅ 模板已生成：", out_path.resolve())
print("请在 Excel 里打开，按下面规则标注 label 列：positive / negative")

from pathlib import Path
import pandas as pd

# 把这个路径改成你实际的文件名，比如：
# Teacher_Cls/en_gold_labeled.xlsx
src = Path("Teacher_Cls/en_gold_labeled.xlsx")

assert src.exists(), f"找不到文件：{src}"

df = pd.read_excel(src)
out_path = src.with_suffix(".csv")   # 变成同名的 .csv
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print("✅ 已转换为 CSV：", out_path.resolve())
print("现在可以直接运行我给你的 Step 2（教师模型训练）代码啦～")

