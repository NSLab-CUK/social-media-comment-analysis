# -*- coding: utf-8 -*-任务一要m2best
"""
bert_train_classifier_minimal.py
用纯 PyTorch 训练 BERT 二分类（兼容任意版本 transformers，无 Trainer/TrainingArguments）
输入：DL_Cls_Out/train.csv, DL_Cls_Out/val.csv  （两列：text,label）
输出：DL_Cls_Out/artifacts/bert_cls_minimal/best  （可直接用于推理）
"""

import os, math, time, json, re
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# ===================== 可改参数 =====================
TRAIN_CSV = "DL_Cls_Out/train.csv"
VAL_CSV   = "DL_Cls_Out/val.csv"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # 可改：bert-base-chinese / chinese-macbert-base 等
MAX_LEN    = 160
EPOCHS     = 3
LR         = 2e-5
TRAIN_BS   = 32
EVAL_BS    = 64
GRAD_ACCUM = 1
FP16       = True
OUT_DIR    = "DL_Cls_Out/artifacts/bert_cls_minimal"
SEED       = 42
PATIENCE   = 2  # 早停
# ===================================================

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor = None

class CSVTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist() if "label" in df.columns else None
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        if self.labels is None:
            return {"text": self.texts[i]}
        return {"text": self.texts[i], "label": self.labels[i]}

class Collator:
    def __init__(self, tokenizer, max_len=160):
        self.tok = tokenizer; self.max_len=max_len
    def __call__(self, samples: List[Dict[str,Any]]):
        texts = [s["text"] for s in samples]
        enc = self.tok(
            texts, truncation=True, padding=True, max_length=self.max_len,
            return_tensors="pt"
        )
        if "label" in samples[0]:
            labels = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)
            return TextBatch(enc["input_ids"], enc["attention_mask"], labels)
        return TextBatch(enc["input_ids"], enc["attention_mask"], None)

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert "text" in df.columns and "label" in df.columns, f"{path} 需要包含列 text,label"
    df = df[["text","label"]].dropna()
    df["text"] = (df["text"].astype(str)
                  .str.replace(r"\s+"," ", regex=True)
                  .str.replace(r"\u200b","", regex=True)
                  .str.strip())
    df["label"] = df["label"].astype(int)
    return df

def evaluate(model, dl, device):
    model.eval()
    probs = []; gold = []
    with torch.no_grad():
        for batch in dl:
            ids = batch.input_ids.to(device)
            mask = batch.attention_mask.to(device)
            y = batch.labels.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            p = torch.softmax(logits, dim=1)[:,1]
            probs += p.cpu().tolist()
            gold  += y.cpu().tolist()
    probs = np.array(probs); gold = np.array(gold)
    pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(gold, pred)
    f1  = f1_score(gold, pred, average="macro")
    return acc, f1

def train():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 数据
    tr_df = load_csv(TRAIN_CSV)
    va_df = load_csv(VAL_CSV)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collate = Collator(tok, MAX_LEN)
    tr_ds = CSVTextDataset(tr_df); va_ds = CSVTextDataset(va_df)
    tr_dl = DataLoader(tr_ds, batch_size=TRAIN_BS, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=EVAL_BS,  shuffle=False, collate_fn=collate)

    # 2) 模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    # 3) 优化器 & 调度器
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay":0.01},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],  "weight_decay":0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)
    num_updates = math.ceil(len(tr_dl)/GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06*num_updates), num_training_steps=num_updates
    )

    scaler = torch.cuda.amp.GradScaler(enabled=FP16 and torch.cuda.is_available())

    # 4) 训练循环
    best_f1 = -1.0
    bad = 0
    for ep in range(1, EPOCHS+1):
        model.train(); running = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        for step, batch in enumerate(tr_dl, start=1):
            ids = batch.input_ids.to(device)
            mask = batch.attention_mask.to(device)
            y = batch.labels.to(device)
            with torch.cuda.amp.autocast(enabled=FP16 and torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, labels=y)
                loss = out.loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            running += loss.item()*GRAD_ACCUM
            if step % 50 == 0:
                print(f"[epoch {ep}] step {step}/{len(tr_dl)} loss={running/step:.4f}")

        acc, f1 = evaluate(model, va_dl, device)
        dt = time.time()-t0
        print(f"[epoch {ep}] val_acc={acc:.4f} val_macroF1={f1:.4f} time={dt:.1f}s")

        if f1 > best_f1:
            best_f1 = f1; bad = 0
            # 保存最优
            save_dir = os.path.join(OUT_DIR, "best")
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            (os.path.join(OUT_DIR,"best_metrics.json") and
             open(os.path.join(OUT_DIR,"best_metrics.json"),"w",encoding="utf-8").write(
                 json.dumps({"val_acc": float(acc), "val_macroF1": float(f1)}, ensure_ascii=False)
             ))
            print(f"✅ 保存最优模型 → {save_dir}")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("[early stop]")
                break

    print("🎉 训练结束，最佳 F1 =", best_f1)

if __name__ == "__main__":
    train()
