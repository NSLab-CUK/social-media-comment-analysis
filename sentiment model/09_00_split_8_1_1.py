# -*- coding: utf-8 -*-情感分析开始
"""
00_split_8_1_1.py
功能：把原始微博文本按 8:1:1 划分为 train/val/test（仅文本，不带情感标签）
输出：
  - DL_Cls_Out/train.raw.csv   （训练用的原始文本，后续再做弱标注/训练）
  - DL_Cls_Out/val.text.csv    （验证集文本）
  - DL_Cls_Out/test.text.csv   （测试集文本）
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== 路径与参数 =====
RAW_CSV   = "weibo_ai_2023_2025_final_clean.csv"      # 你的原始CSV
TEXT_CAND = ["text","内容","文本","微博正文","content","full_text"]  # 自动探测文本列
OUT_DIR   = pathlib.Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42   

def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_CAND:
        if c in df.columns:
            return c
    # 找不到就退回第一列
    return df.columns[0]

def main():
    # 1) 读入与选择文本列
    df0 = pd.read_csv(RAW_CSV, encoding="utf-8", low_memory=False)
    tcol = pick_text_col(df0)
    print(f"[load] 读取：{RAW_CSV} | shape={df0.shape} | 文本列='{tcol}'")

    # 2) 只保留文本列，清洗与去重
    df = df0[[tcol]].rename(columns={tcol: "text"}).dropna()
    df["text"] = (
        df["text"].astype(str)
        .str.replace(r"\s+", " ", regex=True)  # 合并多空白
        .str.replace(r"\u200b", "", regex=True)  # 去零宽空格
        .str.strip()
    )
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[clean] 去重：{before} → {len(df)}")

    # 3) 8:1:1 划分（随机、可复现）
    train_raw, tmp = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
    val_raw, test_raw = train_test_split(tmp, test_size=0.5, random_state=SEED, shuffle=True)

    # 4) 保存
    train_raw.to_csv(OUT_DIR/"train.raw.csv", index=False, encoding="utf-8-sig")
    val_raw.to_csv(OUT_DIR/"val.text.csv", index=False, encoding="utf-8-sig")
    test_raw.to_csv(OUT_DIR/"test.text.csv", index=False, encoding="utf-8-sig")

    # 5) 汇总
    n = len(df)
    print(f"[split] 全量={n} | train={len(train_raw)} ({len(train_raw)/n:.1%}) "
          f"| val={len(val_raw)} ({len(val_raw)/n:.1%}) "
          f"| test={len(test_raw)} ({len(test_raw)/n:.1%})")
    print(f"[save] {OUT_DIR/'train.raw.csv'}")
    print(f"[save] {OUT_DIR/'val.text.csv'}")
    print(f"[save] {OUT_DIR/'test.text.csv'}")
    print("✅ Step 1 完成（仅文本划分）。下一步：在 train.raw.csv 上做情感弱标注。")

if __name__ == "__main__":
    main()
