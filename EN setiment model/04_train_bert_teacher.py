# === Step 1.2: 用人工标注的 800 条训练 BERT 教师模型 ===
import os, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# 配置
SEED       = 42
MODEL_NAME = "bert-base-uncased"   # 够用了
MAX_LEN    = 160
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5
PATIENCE   = 2

BASE_DIR = Path("Teacher_Cls")
IN_CSV   = BASE_DIR / "en_gold_labeled.csv"
OUT_PATH = BASE_DIR / "bert_teacher_model.pt"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

assert IN_CSV.exists(), f"找不到人工标注文件：{IN_CSV}"
df = pd.read_csv(IN_CSV)

# 只保留 positive / negative
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["positive","negative"])].copy()
assert not df.empty, "标注文件为空或没有 positive/negative"

# 选文本列
text_col = "text_raw" if "text_raw" in df.columns else (
           "text_norm" if "text_norm" in df.columns else None)
assert text_col is not None, "需要 text_raw 或 text_norm 列"

print("样本数：", len(df))
print("label 分布：", df["label"].value_counts())

# 切 train/dev
train_df, dev_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

labels = sorted(train_df["label"].astype(str).unique().tolist())
label2id = {c:i for i,c in enumerate(labels)}
id2label = {i:c for c,i in label2id.items()}
num_classes = len(labels)
print("标签集合：", labels)

# 类别权重（防止不平衡）
freq = train_df["label"].value_counts()
weights = [len(train_df) / (len(freq) * freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class BertDS(Dataset):
    def __init__(self, df, text_col, label2id, tokenizer, max_len=128):
        self.texts  = df[text_col].astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df["label"].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_len   = max_len
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

ds_tr  = BertDS(train_df, text_col, label2id, tokenizer, MAX_LEN)
ds_dev = BertDS(dev_df,   text_col, label2id, tokenizer, MAX_LEN)
dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False)

class BertTeacher(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:,0]
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = BertTeacher(MODEL_NAME, num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1
)

def run_epoch(loader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ys, ps = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds==labels).sum().item()
        total += labels.size(0)
        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())
    acc = total_correct / total if total>0 else 0.0
    return total_loss/total, acc, np.array(ys), np.array(ps)

best_f1, best_state, wait = -1.0, None, 0

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _ = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)
    print(f"[Teacher Ep {ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | dev {dv_loss:.4f}/{dv_acc:.4f} F1={dv_f1:.4f}")

    if dv_f1 > best_f1 + 1e-4:
        best_f1, wait = dv_f1, 0
        best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stop, best dev F1 = {best_f1:.4f}")
            break

if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

# 在 dev 上打印详细报告，自己心里有数
model.eval()
ys, ps = [], []
with torch.no_grad():
    for batch in dl_dev:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_tensor = batch["labels"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)
        ys.extend(labels_tensor.cpu().tolist())
        ps.extend(preds.cpu().tolist())

print("\n=== TEACHER DEV REPORT ===")
print(classification_report(ys, ps, target_names=labels, digits=3))
print("Confusion Matrix:\n", confusion_matrix(ys, ps))

# 保存 teacher 模型
torch.save({
    "model_name": MODEL_NAME,
    "state_dict": model.state_dict(),
    "labels": labels,
    "label2id": label2id,
}, OUT_PATH)

print("\n✅ 教师模型已保存到：", OUT_PATH.resolve())
