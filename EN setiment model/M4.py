# ===== step7_bert_trans_seq.py：BERT 编码 → TransformerEncoder =====m4
import os, json, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModel

# ===== 配置 =====
SEED        = 42
MODEL_NAME  = "bert-base-uncased"
FREEZE_BERT = True            # 冻结 BERT，只训练上面的 Transformer + 分类头
MAX_LEN     = 160
BATCH_SIZE  = 8
EPOCHS      = 5
LR          = 2e-4
PATIENCE    = 2

IN_DIR  = Path("DL_Cls_In")
OUT_DIR = Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ===== 读数据 & 标签 =====
train = pd.read_csv(IN_DIR/"train.csv")
dev   = pd.read_csv(IN_DIR/"dev.csv")
test  = pd.read_csv(IN_DIR/"test.csv")

text_col = "text_raw" if "text_raw" in train.columns else ("text_norm" if "text_norm" in train.columns else None)
assert text_col is not None, "需要 text_raw 或 text_norm 列"
assert "label" in train.columns

labels = sorted(train["label"].astype(str).unique().tolist())
label2id = {c:i for i,c in enumerate(labels)}
id2label = {i:c for c,i in label2id.items()}
num_classes = len(labels)

freq = train["label"].value_counts()
weights = [len(train)/(len(freq)*freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float)

# ===== Dataset & DataLoader =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class BertSeqDataset(Dataset):
    def __init__(self, df, text_col, label2id, tokenizer, max_len):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df["label"].astype(str).tolist()]
        self.tokenizer = tokenizer; self.max_len=max_len
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

ds_tr  = BertSeqDataset(train, text_col, label2id, tokenizer, MAX_LEN)
ds_dev = BertSeqDataset(dev,   text_col, label2id, tokenizer, MAX_LEN)
ds_te  = BertSeqDataset(test,  text_col, label2id, tokenizer, MAX_LEN)

dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False)
dl_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False)

# ===== 模型：BERT + TransformerEncoder + max-pool =====
class BertTransClassifier(nn.Module):
    def __init__(self, model_name, num_classes, nhead=8, nlayers=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state          # [B, L, H]
        key_padding_mask = (attention_mask == 0)   # True 表示 padding
        enc_out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        pooled,_ = torch.max(enc_out, dim=1)       # max-pool over time
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = BertTransClassifier(MODEL_NAME, num_classes).to(DEVICE)

if FREEZE_BERT:
    for p in model.bert.parameters():
        p.requires_grad = False

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

# ===== 训练 & 早停 =====
def run_epoch(dataloader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ys, ps = [], []
    for batch in dataloader:
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
        total_loss += loss.item()*labels.size(0)
        total_correct += (preds==labels).sum().item()
        total += labels.size(0)
        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())
    acc = total_correct/total if total>0 else 0.0
    return total_loss/total, acc, np.array(ys), np.array(ps)

best_f1, best_state, wait = -1.0, None, 0
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _      = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)
    print(f"[bert_trans Ep{ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | dev {dv_loss:.4f}/{dv_acc:.4f} F1={dv_f1:.4f}")
    if dv_f1 > best_f1 + 1e-4:
        best_f1, wait = dv_f1, 0
        best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stop, best dev F1 =", best_f1)
            break

if best_state is not None:
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

# ===== 评估 & 保存 =====
def eval_and_dump(name, df, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)
            ys.extend(labels.cpu().tolist())
            ps.extend(preds.cpu().tolist())

    ys = np.array(ys); ps = np.array(ps)
    cls_ids = sorted(set(ys) | set(ps))
    target_names = [id2label[i] for i in cls_ids]

    print(f"\n=== {name.upper()} (bert_trans) ===")
    print(classification_report(ys, ps, labels=cls_ids, target_names=target_names, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(ys, ps, labels=cls_ids))

    out = df.copy()
    out["pred"] = [id2label[i] for i in ps]
    out.to_csv(OUT_DIR/f"preds_bert_trans_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"→ 保存: DL_Cls_Out/preds_bert_trans_{name}.csv")

eval_and_dump("dev", dev, dl_dev)
eval_and_dump("test", test, dl_te)

torch.save({
    "model_type": "bert_trans",
    "model_name": MODEL_NAME,
    "state_dict": model.state_dict(),
    "labels": labels
}, OUT_DIR/"bert_trans_model.pt")
print("✅ bert_trans 完成。")
