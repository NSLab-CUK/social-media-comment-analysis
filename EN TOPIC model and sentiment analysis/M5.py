# step6_w2v_seq_models.py m5
import os, re, json, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from gensim.models import Word2Vec

# ===== 配置 =====
SEED        = 42
MODEL_TYPE  = "w2v_lstm"      # "w2v_lstm" 或 "w2v_trans"
EMB_DIM     = 200
HIDDEN      = 256
BATCH_SIZE  = 128
EPOCHS      = 15
LR          = 2e-3
PATIENCE    = 3
MAX_LEN     = 160
MIN_COUNT   = 2
IN_DIR      = Path("DL_Cls_In")
OUT_DIR     = Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

train = pd.read_csv(IN_DIR/"train.csv")
dev   = pd.read_csv(IN_DIR/"dev.csv")
test  = pd.read_csv(IN_DIR/"test.csv")
text_col = "text_raw" if "text_raw" in train.columns else ("text_norm" if "text_norm" in train.columns else None)
assert text_col is not None

labels = sorted(train["label"].astype(str).unique().tolist())
label2id = {c:i for i,c in enumerate(labels)}
id2label = {i:c for c,i in label2id.items()}
num_classes = len(labels)

freq = train["label"].value_counts()
weights = [len(train)/(len(freq)*freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float)

_tok_pat = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]|[0-9]+|[^\s\w]", re.UNICODE)
def tokenize(s: str):
    s = str(s).lower().strip()
    return _tok_pat.findall(s)

# ===== 训练 Word2Vec =====
sentences = [tokenize(s) for s in train[text_col].astype(str).tolist()]
w2v = Word2Vec(
    sentences,
    vector_size=EMB_DIM,
    window=5,
    min_count=MIN_COUNT,
    workers=4,
    sg=1,
    epochs=10,
    seed=SEED
)

# 构建 vocab: 特殊符号 + w2v 词表
stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
for w in w2v.wv.index_to_key:
    if w not in stoi:
        stoi[w] = len(stoi)
itos = {i:w for w,i in stoi.items()}
vocab_size = len(stoi)

# 初始化 embedding 权重
emb_weight = torch.randn(vocab_size, EMB_DIM) * 0.01
for word, idx in stoi.items():
    if word in w2v.wv:
        emb_weight[idx] = torch.tensor(w2v.wv[word])

def encode_text(s, max_len=MAX_LEN):
    toks = ["<bos>"] + tokenize(s) + ["<eos>"]
    ids = [stoi.get(t, stoi["<unk>"]) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

class TxtDS(Dataset):
    def __init__(self, df):
        self.X = [encode_text(s) for s in df[text_col].astype(str).tolist()]
        self.y = [label2id[str(l)] for l in df["label"].astype(str).tolist()]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.as_tensor(self.X[i]), torch.as_tensor(self.y[i])

ds_tr, ds_dev, ds_te = TxtDS(train), TxtDS(dev), TxtDS(test)
dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False)
dl_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False)

class W2V_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, hidden=HIDDEN, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb.weight.data.copy_(emb_weight)
        self.lstm = nn.LSTM(emb_dim, hidden//2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        emb = self.emb(x)             # [B,L,E]
        out,_ = self.lstm(emb)        # [B,L,H]
        pooled,_ = torch.max(out, dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits

class W2V_Trans(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, nhead=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb.weight.data.copy_(emb_weight)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        emb = self.emb(x)             # [B,L,E]
        # padding 为 0 的位置 mask 掉
        key_padding_mask = (x == 0)
        out = self.encoder(emb, src_key_padding_mask=key_padding_mask)
        pooled,_ = torch.max(out, dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if MODEL_TYPE == "w2v_lstm":
    model = W2V_LSTM(vocab_size, EMB_DIM, num_classes)
elif MODEL_TYPE == "w2v_trans":
    model = W2V_Trans(vocab_size, EMB_DIM, num_classes)
else:
    raise ValueError("MODEL_TYPE 只能是 w2v_lstm 或 w2v_trans")

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

def run_epoch(dataloader, train_mode=True):
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ys, ps = [], []
    for X,y in dataloader:
        X = X.to(DEVICE); y = y.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(X)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item()*X.size(0)
        total_correct += (preds==y).sum().item()
        total += X.size(0)
        ys.extend(y.cpu().tolist()); ps.extend(preds.cpu().tolist())
    acc = total_correct/total if total>0 else 0.0
    return total_loss/total, acc, np.array(ys), np.array(ps)

best_f1, best_state, wait = -1.0, None, 0
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _ = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)
    print(f"[{MODEL_TYPE} Ep{ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | dev {dv_loss:.4f}/{dv_acc:.4f} F1={dv_f1:.4f}")
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

def eval_and_dump(name, df, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X,y in loader:
            X = X.to(DEVICE); y = y.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)
            ys.extend(y.cpu().tolist()); ps.extend(preds.cpu().tolist())
    print(f"\n=== {name.upper()} ({MODEL_TYPE}) ===")
    print(classification_report(ys, ps, target_names=labels, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(ys, ps))
    out = df.copy()
    out["pred"] = [id2label[i] for i in ps]
    out.to_csv(OUT_DIR/f"preds_{MODEL_TYPE}_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"→ 保存: DL_Cls_Out/preds_{MODEL_TYPE}_{name}.csv")

eval_and_dump("dev", dev, dl_dev)
eval_and_dump("test", test, dl_te)

torch.save({
    "model_type": MODEL_TYPE,
    "state_dict": model.state_dict(),
    "labels": labels,
}, OUT_DIR/f"{MODEL_TYPE}_model.pt")

print("✅", MODEL_TYPE, "完成。")
