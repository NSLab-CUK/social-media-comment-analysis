# step4_bert_cls.py M2
import os, re, json, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from transformers import AutoTokenizer, AutoModel

# ========= 配置 =========
SEED        = 42
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 160
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5
PATIENCE    = 2   # 早停轮数
IN_DIR      = Path("DL_Cls_In")
OUT_DIR     = Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 随机种子 =========
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ========= 读数据 =========
train = pd.read_csv(IN_DIR/"train.csv")
dev   = pd.read_csv(IN_DIR/"dev.csv")
test  = pd.read_csv(IN_DIR/"test.csv")

text_col = "text_raw" if "text_raw" in train.columns else (
    "text_norm" if "text_norm" in train.columns else None
)
assert text_col is not None, "需要 text_raw 或 text_norm 列"
assert "label" in train.columns, "缺少 label 列"

# 标签映射（注意这里用 label_names，避免和 sklearn 的 labels 参数撞名）
label_names = sorted(train["label"].astype(str).unique().tolist())
label2id = {c:i for i,c in enumerate(label_names)}
id2label = {i:c for c,i in label2id.items()}
num_classes = len(label_names)

with open(OUT_DIR/"label2id_bert.json","w",encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)

# 类别权重（平衡不均衡）
freq = train["label"].value_counts()
class_weights_list = [len(train) / (len(freq) * freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights_list, dtype=torch.float)

# ========= Dataset & Dataloader =========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class BertDataset(Dataset):
    def __init__(self, df, text_col, label2id, tokenizer, max_len=128):
        self.texts  = df[text_col].astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df["label"].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
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

ds_tr  = BertDataset(train, text_col, label2id, tokenizer, MAX_LEN)
ds_dev = BertDataset(dev,   text_col, label2id, tokenizer, MAX_LEN)
ds_te  = BertDataset(test,  text_col, label2id, tokenizer, MAX_LEN)

dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False)
dl_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False)

# ========= 模型：BERT + 线性分类头 =========
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 用 pooler_output（如果没有就用 CLS）
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output             # [B, H]
        else:
            pooled = out.last_hidden_state[:, 0]   # [B, H] CLS
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = BertClassifier(MODEL_NAME, num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1
)

# ========= 训练 & 评估 =========
def run_epoch(dataloader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ys, ps = [], []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch   = batch["labels"].to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss   += loss.item() * labels_batch.size(0)
        total_correct += (preds == labels_batch).sum().item()
        total        += labels_batch.size(0)
        ys.extend(labels_batch.cpu().tolist())
        ps.extend(preds.cpu().tolist())

    acc = total_correct / total if total > 0 else 0.0
    return total_loss/total, acc, np.array(ys), np.array(ps)

best_f1, best_state, wait = -1.0, None, 0

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _ = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)

    print(f"[BERT_CLS Ep {ep:02d}] "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"dev loss {dv_loss:.4f} acc {dv_acc:.4f} F1 {dv_f1:.4f}")

    if dv_f1 > best_f1 + 1e-4:
        best_f1 = dv_f1
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stop at epoch {ep}, best dev macro-F1 = {best_f1:.4f}")
            break

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

def eval_and_dump(name, df, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch   = batch["labels"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds  = logits.argmax(dim=1)
            ys.extend(labels_batch.cpu().tolist())
            ps.extend(preds.cpu().tolist())

    ys = np.array(ys); ps = np.array(ps)

    # 显式指定所有类别 id 与名字，避免“3 类 / 8 名字”报错
    all_ids      = list(range(num_classes))
    target_names = [id2label[i] for i in all_ids]

    print(f"\n=== {name.upper()} (BERT CLS) ===")
    print(classification_report(
        ys, ps,
        labels=all_ids,
        target_names=target_names,
        digits=3,
        zero_division=0
    ))
    print("Confusion Matrix:\n", confusion_matrix(ys, ps, labels=all_ids))

    out = df.copy()
    out["pred"] = [id2label[i] for i in ps]
    out.to_csv(OUT_DIR/f"preds_bert_cls_{name}.csv",
               index=False, encoding="utf-8-sig")
    print(f"→ 保存: DL_Cls_Out/preds_bert_cls_{name}.csv")

eval_and_dump("dev", dev, dl_dev)
eval_and_dump("test", test, dl_te)

torch.save({
    "model_name": MODEL_NAME,
    "state_dict": model.state_dict(),
    "label2id": label2id,
}, OUT_DIR/"bert_cls_model.pt")
print("✅ BERT CLS 分类完成。")
