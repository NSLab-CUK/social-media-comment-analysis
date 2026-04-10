# ===== step2_cn_tfidf_lr.py：中文 TF-IDF + Logistic Regression baseline（与英文版同方法/同输出风格）=====M1
from pathlib import Path
import re
import pandas as pd
import numpy as np

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

OUT_DIR = Path("DL_Cls_Out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1) 自动选择中文数据路径（兼容你旧中文代码 & 新英文pipeline）
# ---------------------------
# 旧中文：DL_Cls_Out/train.csv, DL_Cls_Out/val.csv, DL_Cls_Out/test.text.csv
# 新英文：DL_Cls_In/train.csv, DL_Cls_In/dev.csv, DL_Cls_In/test.csv
CN_TRAIN_A = Path("DL_Cls_Out/train.csv")
CN_DEV_A   = Path("DL_Cls_Out/val.csv")
CN_TEST_A  = Path("DL_Cls_Out/test.text.csv")

CN_TRAIN_B = Path("DL_Cls_In/train.csv")
CN_DEV_B   = Path("DL_Cls_In/dev.csv")
CN_TEST_B  = Path("DL_Cls_In/test.csv")

def pick_paths():
    if CN_TRAIN_A.exists() and CN_DEV_A.exists() and CN_TEST_A.exists():
        return CN_TRAIN_A, CN_DEV_A, CN_TEST_A, "A(旧中文DL_Cls_Out)"
    if CN_TRAIN_B.exists() and CN_DEV_B.exists() and CN_TEST_B.exists():
        return CN_TRAIN_B, CN_DEV_B, CN_TEST_B, "B(新pipeline DL_Cls_In)"
    raise FileNotFoundError(
        "找不到中文数据文件。\n"
        "需要以下任一组：\n"
        "A) DL_Cls_Out/train.csv + DL_Cls_Out/val.csv + DL_Cls_Out/test.text.csv\n"
        "B) DL_Cls_In/train.csv + DL_Cls_In/dev.csv + DL_Cls_In/test.csv"
    )

TRAIN_PATH, DEV_PATH, TEST_PATH, WHICH = pick_paths()
print(f"[CN-M1] Using dataset paths: {WHICH}")
print("  train:", TRAIN_PATH)
print("  dev  :", DEV_PATH)
print("  test :", TEST_PATH)

train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig" if "utf" in "utf-8-sig" else None)
dev   = pd.read_csv(DEV_PATH,   encoding="utf-8-sig" if "utf" in "utf-8-sig" else None)
test  = pd.read_csv(TEST_PATH,  encoding="utf-8-sig" if "utf" in "utf-8-sig" else None)

# ---------------------------
# 2) 自动识别文本列 & 标签列
# ---------------------------
def find_text_col(df):
    # 中文旧版常见：text
    # 新版：text_raw / text_norm
    for c in ["text", "text_raw", "text_norm"]:
        if c in df.columns:
            return c
    # 兜底：如果只有一列就当作文本
    if len(df.columns) == 1:
        return df.columns[0]
    return None

def find_label_col(df):
    for c in ["label", "y", "sentiment"]:
        if c in df.columns:
            return c
    return None

text_col = find_text_col(train)
label_col = find_label_col(train)

assert text_col is not None, f"[CN-M1] 训练集找不到文本列：需要 text / text_raw / text_norm（或只有一列）"
assert label_col is not None, f"[CN-M1] 训练集找不到标签列：需要 label（或 y/sentiment）"

# dev/test 也尽量对齐
dev_text_col  = find_text_col(dev)  or text_col
test_text_col = find_text_col(test) or text_col

# test 可能没有 label（比如 test.text.csv）
dev_label_col  = find_label_col(dev)
test_label_col = find_label_col(test)

# ---------------------------
# 3) 中文清洗 + 分词（jieba）
# ---------------------------
def clean_cn(s: str) -> str:
    s = str(s)
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jieba_tokenize(s: str):
    s = clean_cn(s)
    # jieba.lcut 返回 list[str]
    return [t for t in jieba.lcut(s) if t.strip()]

# TF-IDF 直接用 tokenizer（注意：传 tokenizer 时，TfidfVectorizer 自己会做 preprocessor/lowercase）
# 中文不需要 lowercase
vectorizer = TfidfVectorizer(
    tokenizer=jieba_tokenize,
    token_pattern=None,          # ✅ 关闭默认 token_pattern，避免与 tokenizer 冲突
    ngram_range=(1,2),
    max_features=100_000,
    min_df=2,
    max_df=0.95,
)

clf = LogisticRegression(
    max_iter=2000,
    n_jobs=1,
    C=4.0,
    class_weight="balanced"      # 与你英文一致
)

pipe = Pipeline([
    ("tfidf", vectorizer),
    ("clf", clf)
])

# ---------------------------
# 4) 训练
# ---------------------------
X_tr = train[text_col].astype(str).map(clean_cn)
y_tr = train[label_col].astype(str)  # 保持 label 为字符串更稳（和英文一致）

pipe.fit(X_tr, y_tr)

# ---------------------------
# 5) 评估 + 保存（同英文版风格）
# ---------------------------
def eval_and_dump(split_name, df_split, x_col, y_col=None):
    X = df_split[x_col].astype(str).map(clean_cn)
    y_true = df_split[y_col].astype(str).tolist() if (y_col is not None and y_col in df_split.columns) else None

    y_pred = pipe.predict(X)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X)
    except Exception:
        pass

    print(f"\n=== {split_name.upper()} (CN-M1: TFIDF+LR) ===")
    if y_true is not None:
        print(classification_report(y_true, y_pred, digits=3))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    else:
        print("(No ground-truth labels found in this split; only dumping predictions.)")

    out = df_split.copy()
    out["pred"] = y_pred

    if y_proba is not None:
        classes = pipe.named_steps["clf"].classes_.tolist()
        for i, c in enumerate(classes):
            out[f"proba_{c}"] = y_proba[:, i]

    pred_path = OUT_DIR / f"preds_lr_cn_{split_name}.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"→ 预测已保存: {pred_path}")

    # 误差样例（按最大置信度排序）——只有有标签才做
    if y_true is not None and y_proba is not None:
        conf = y_proba.max(axis=1)
        errs = out[out["pred"].astype(str) != out[y_col].astype(str)].copy()
        if len(errs) > 0:
            # conf 对应原行序，需要用 mask 取对应值
            mask = (out["pred"].astype(str) != out[y_col].astype(str)).to_numpy()
            errs["max_conf"] = conf[mask]
            errs = errs.sort_values("max_conf", ascending=False).head(50)
            err_path = OUT_DIR / f"errors_top50_cn_{split_name}.csv"
            errs.to_csv(err_path, index=False, encoding="utf-8-sig")
            print(f"→ 已保存误差样例: {err_path}")

# dev/test
eval_and_dump("dev", dev, dev_text_col, dev_label_col)
eval_and_dump("test", test, test_text_col, test_label_col)

print("✅ 中文 M1 基线完成：输出在 DL_Cls_Out/")
