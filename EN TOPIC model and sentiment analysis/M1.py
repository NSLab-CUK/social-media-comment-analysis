# STEP 2: TF-IDF + Logistic Regression baseline M1
from pathlib import Path
Path("DL_Cls_Out").mkdir(parents=True, exist_ok=True)

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IN_DIR = Path("DL_Cls_In")
assert (IN_DIR/"train.csv").exists(), "找不到 DL_Cls_In/train.csv，请先运行切分步骤。"

train = pd.read_csv(IN_DIR/"train.csv")
dev   = pd.read_csv(IN_DIR/"dev.csv")
test  = pd.read_csv(IN_DIR/"test.csv")

# 自动识别文本列
text_col = "text_raw" if "text_raw" in train.columns else ("text_norm" if "text_norm" in train.columns else None)
assert text_col is not None, "需要 text_raw 或 text_norm 列"

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=100_000,
                              min_df=2, max_df=0.95, lowercase=True)),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=1, C=4.0, class_weight="balanced"))
])

pipe.fit(train[text_col], train["label"])

def eval_and_dump(split_name, df_split):
    y_true = df_split["label"].tolist()
    y_pred = pipe.predict(df_split[text_col])
    y_proba = None
    try:
        y_proba = pipe.predict_proba(df_split[text_col])
    except Exception:
        pass

    print(f"\n=== {split_name.upper()} ===")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # 保存预测与误差样例
    out = df_split.copy()
    out["pred"] = y_pred
    if y_proba is not None:
        classes = pipe.named_steps["clf"].classes_.tolist()
        proba_cols = {f"proba_{c}": y_proba[:,i] for i,c in enumerate(classes)}
        for k,v in proba_cols.items(): out[k] = v
    out.to_csv(f"DL_Cls_Out/preds_lr_{split_name}.csv", index=False, encoding="utf-8-sig")

    # 误差样例（按最大置信度排序）
    Path("DL_Cls_Out").mkdir(parents=True, exist_ok=True)
    if y_proba is not None:
        conf = y_proba.max(axis=1)
        errs = out[out["pred"]!=out["label"]].copy()
        errs["max_conf"] = conf[out.index[out["pred"]!=out["label"]]]
        errs = errs.sort_values("max_conf", ascending=False).head(50)
        errs.to_csv(f"DL_Cls_Out/errors_top50_{split_name}.csv", index=False, encoding="utf-8-sig")
        print(f"→ 已保存误差样例: DL_Cls_Out/errors_top50_{split_name}.csv")
    print(f"→ 预测已保存: DL_Cls_Out/preds_lr_{split_name}.csv")

# 评估 dev / test
eval_and_dump("dev", dev)
eval_and_dump("test", test)
print("✅ 基线完成：输出在 DL_Cls_Out/")
