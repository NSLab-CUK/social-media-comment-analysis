# step5_bert_seq_models.pyM3
import os, json, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModel

# ===== 配置 =====q
SEED        = 42
MODEL_NAME  = "bert-base-uncased"
MODEL_TYPE  = "bert_lstm"     # "bert_lstm" 或 "bert_trans"
FREEZE_BERT = True            # True: 只用 BERT 做固定编码
MAX_LEN     = 160
BATCH_SIZE  = 8               # 序列模型 batch 小一点
EPOCHS      = 5
LR          = 2e-4            # 冻结 BERT 时可以稍大
PATIENCE    = 2
IN_DIR      = Path("DL_Cls_In")
OUT_DIR     = Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ===== 读数据 =====
train = pd.read_csv(IN_DIR/"train.csv")
dev   = pd.read_csv(IN_DIR/"dev.csv")
test  = pd.read_csv(IN_DIR/"test.csv")

text_col = "text_raw" if "text_raw" in train.columns else (
    "text_norm" if "text_norm" in train.columns else None
)
assert text_col is not None, "需要 text_raw 或 text_norm 列"

# 标签映射
label_names = sorted(train["label"].astype(str).unique().tolist())
label2id = {c:i for i,c in enumerate(label_names)}
id2label = {i:c for c,i in label2id.items()}
num_classes = len(label_names)

freq = train["label"].value_counts()
class_weights_list = [len(train)/(len(freq)*freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights_list, dtype=torch.float)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===== Dataset / Dataloader =====
class BertSeqDataset(Dataset):
    def __init__(self, df, text_col, label2id, tokenizer, max_len):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label2id[str(x)] for x in df["label"].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len
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

# ===== 模型定义：BERT + LSTM / Transformer =====
class BertLSTMClassifier(nn.Module):
    def __init__(self, model_name, num_classes,
                 lstm_hidden=384, num_layers=1,
                 bidirectional=True, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state              # [B, L, H]

        # 真实长度（按 attention_mask 求和）
        lengths = attention_mask.sum(dim=1)      # [B]
        lengths_cpu = lengths.cpu()

        # pack -> LSTM -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out_seq, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )                                         # [B, L', H']

        # 基于 mask 做 max pooling（忽略 padding）
        max_len = out_seq.size(1)
        mask = (torch.arange(max_len, device=out_seq.device)[None, :] <
                lengths[:, None])                # [B, L]
        mask = mask.unsqueeze(-1)                # [B, L, 1]
        out_seq = out_seq.masked_fill(~mask, -1e9)
        pooled = out_seq.max(dim=1).values       # [B, H']

        logits = self.fc(self.dropout(pooled))
        return logits

class BertTransClassifier(nn.Module):
    def __init__(self, model_name, num_classes,
                 nhead=8, nlayers=2, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state               # [B, L, H]

        # TransformerEncoder 里：True 表示要 mask 掉
        key_padding_mask = (attention_mask == 0)  # [B, L]
        enc_out = self.encoder(
            seq,
            src_key_padding_mask=key_padding_mask
        )                                         # [B, L, H]
        pooled, _ = torch.max(enc_out, dim=1)     # [B, H]
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_TYPE == "bert_lstm":
    model = BertLSTMClassifier(MODEL_NAME, num_classes)
elif MODEL_TYPE == "bert_trans":
    model = BertTransClassifier(MODEL_NAME, num_classes)
else:
    raise ValueError("MODEL_TYPE 只能是 bert_lstm 或 bert_trans")

# 冻结 / 不冻结 BERT
if FREEZE_BERT:
    for p in model.bert.parameters():
        p.requires_grad = False

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1
)

# ===== 训练循环 =====
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
    tr_loss, tr_acc, _, _    = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)

    print(f"[{MODEL_TYPE} Ep{ep:02d}] "
          f"train {tr_loss:.4f}/{tr_acc:.4f} | "
          f"dev {dv_loss:.4f}/{dv_acc:.4f} F1={dv_f1:.4f}")

    if dv_f1 > best_f1 + 1e-4:
        best_f1, wait = dv_f1, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stop, best dev F1 =", best_f1)
            break

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# ===== 评估 & 打印（修好 labels 部分） =====
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

    all_ids      = list(range(num_classes))
    target_names = [id2label[i] for i in all_ids]

    print(f"\n=== {name.upper()} ({MODEL_TYPE}) ===")
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
    out.to_csv(OUT_DIR/f"preds_{MODEL_TYPE}_{name}.csv",
               index=False, encoding="utf-8-sig")
    print(f"→ 保存: DL_Cls_Out/preds_{MODEL_TYPE}_{name}.csv")

eval_and_dump("dev", dev, dl_dev)
eval_and_dump("test", test, dl_te)

torch.save({
    "model_type": MODEL_TYPE,
    "model_name": MODEL_NAME,
    "state_dict": model.state_dict(),
    "label2id": label2id,
}, OUT_DIR/f"{MODEL_TYPE}_model.pt")
print("✅", MODEL_TYPE, "完成。")
