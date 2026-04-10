# ===== step8_w2v_trans_fixed.py：Word2Vec 编码 → TransformerEncoder m6
import os, re, json, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---- 配置 ----
SEED        = 42
EMB_DIM     = 300
MAX_LEN     = 160
BATCH_SIZE  = 128
EPOCHS      = 20
LR          = 3e-4           # ✅ 降低学习率（原 2e-3）
PATIENCE    = 4              # ✅ 稍微放宽早停
MIN_FREQ    = 2
MAX_VOCAB   = 50000
W2V_EPOCHS  = 10
FREEZE_W2V  = False          # ✅ True: 先完全冻结 embedding（更稳）；False: 允许微调
CLIP_NORM   = 1.0            # ✅ 梯度裁剪
WEIGHT_DECAY= 0.01

IN_DIR  = Path("DL_Cls_In")
OUT_DIR = Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ---- 读数据 & 标签 ----
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

# 类别权重（保持你原逻辑，但注意：如果仍塌缩，可临时注释掉权重试跑）
freq = train["label"].value_counts()
weights = [len(train)/(len(freq)*freq[id2label[i]]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float)

# ---- tokenizer & vocab ----
_tok_pat = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]|[0-9]+|[^\s\w]", re.UNICODE)
def tokenize(s: str):
    s = str(s).lower().strip()
    return _tok_pat.findall(s)

def build_vocab(texts, min_freq=MIN_FREQ, max_vocab=MAX_VOCAB):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    items = [(w,f) for w,f in counter.items() if f>=min_freq]
    items.sort(key=lambda x:(-x[1], x[0]))
    if max_vocab is not None:
        items = items[:max_vocab-4]
    stoi = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
    for w,_ in items:
        if w not in stoi:
            stoi[w] = len(stoi)
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos

stoi, itos = build_vocab(train[text_col].tolist(), MIN_FREQ, MAX_VOCAB)
vocab_size = len(stoi)

def encode_text(s, max_len=MAX_LEN):
    toks = ["<bos>"] + tokenize(s) + ["<eos>"]
    ids  = [stoi.get(t, stoi["<unk>"]) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]]*(max_len-len(ids))
    return np.array(ids, dtype=np.int64)

# ---- 训练 Word2Vec ----
sentences = [tokenize(s) for s in train[text_col].astype(str).tolist()]
w2v = Word2Vec(
    sentences,
    vector_size=EMB_DIM,
    window=5,
    min_count=MIN_FREQ,
    workers=4,
    sg=1,
    seed=SEED,
    epochs=W2V_EPOCHS,
)

# 构造 embedding matrix
emb_matrix = np.random.normal(scale=0.01, size=(vocab_size, EMB_DIM)).astype(np.float32)
hit = 0
for idx, token in itos.items():
    if token in w2v.wv:
        emb_matrix[idx] = w2v.wv[token]
        hit += 1
print(f"[w2v] vocab={vocab_size}, hit_in_w2v={hit}, hit_rate={hit/max(vocab_size,1):.3f}")

# ---- Dataset & DataLoader ----
class TxtDS(Dataset):
    def __init__(self, df):
        self.X = [encode_text(s) for s in df[text_col].astype(str).tolist()]
        self.y = [label2id[str(l)] for l in df["label"].astype(str).tolist()]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.as_tensor(self.X[i]), torch.as_tensor(self.y[i])

ds_tr, ds_dev, ds_te = TxtDS(train), TxtDS(dev), TxtDS(test)
dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
dl_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---- 位置编码（sin-cos）----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---- 模型：Word2Vec Embedding + TransformerEncoder（修复版）----
class W2VTransClassifier(nn.Module):
    def __init__(self, emb_matrix, num_classes, nhead=6, nlayers=2, dropout=0.2):
        super().__init__()
        num_embeddings, emb_dim = emb_matrix.shape

        self.emb = nn.Embedding(num_embeddings, emb_dim, padding_idx=0)
        self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))

        # ✅ 可选：冻结词向量更稳（先跑稳再解冻）
        self.emb.weight.requires_grad = (not FREEZE_W2V)

        self.pos = PositionalEncoding(emb_dim, max_len=MAX_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, L]
        pad_mask = (x == 0)                     # [B, L] True for PAD
        emb = self.emb(x)                       # [B, L, E]
        emb = self.pos(emb)                     # ✅ 加位置编码
        enc_out = self.encoder(emb, src_key_padding_mask=pad_mask)  # [B, L, E]

        # ✅ masked mean pooling（比 max pooling 稳）
        keep = (~pad_mask).unsqueeze(-1).float()     # [B, L, 1]
        enc_out = enc_out * keep
        pooled = enc_out.sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = W2VTransClassifier(emb_matrix, num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

# ---- 训练 & 早停 ----
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
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)  # ✅ 梯度裁剪
                optimizer.step()
        pred = logits.argmax(dim=1)
        total_loss += loss.item()*X.size(0)
        total_correct += (pred==y).sum().item()
        total += X.size(0)
        ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
    acc = total_correct/total if total>0 else 0.0
    return total_loss/total, acc, np.array(ys), np.array(ps)

best_f1, best_state, wait = -1.0, None, 0
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _        = run_epoch(dl_tr, True)
    dv_loss, dv_acc, dv_y, dv_p  = run_epoch(dl_dev, False)
    dv_f1 = f1_score(dv_y, dv_p, average="macro")
    scheduler.step(dv_f1)
    print(f"[w2v_trans_fix Ep{ep:02d}] train {tr_loss:.4f}/{tr_acc:.4f} | dev {dv_loss:.4f}/{dv_acc:.4f} F1={dv_f1:.4f}")

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

# ---- 评估 & 保存 ----
def eval_and_dump(name, df, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X,y in loader:
            X = X.to(DEVICE); y = y.to(DEVICE)
            logits = model(X)
            pred = logits.argmax(dim=1)
            ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())

    ys = np.array(ys); ps = np.array(ps)
    cls_ids = sorted(set(ys) | set(ps))
    target_names = [id2label[i] for i in cls_ids]

    print(f"\n=== {name.upper()} (w2v_trans_fix) ===")
    print(classification_report(ys, ps, labels=cls_ids, target_names=target_names, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(ys, ps, labels=cls_ids))

    out = df.copy()
    out["pred"] = [id2label[i] for i in ps]
    out.to_csv(OUT_DIR/f"preds_w2v_trans_fix_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"→ 保存: DL_Cls_Out/preds_w2v_trans_fix_{name}.csv")

eval_and_dump("dev", dev, dl_dev)
eval_and_dump("test", test, dl_te)

torch.save({
    "model_type": "w2v_trans_fix",
    "state_dict": model.state_dict(),
    "labels": labels,
    "config": {
        "EMB_DIM": EMB_DIM, "MAX_LEN": MAX_LEN, "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS, "LR": LR, "PATIENCE": PATIENCE, "MIN_FREQ": MIN_FREQ,
        "MAX_VOCAB": MAX_VOCAB, "W2V_EPOCHS": W2V_EPOCHS, "FREEZE_W2V": FREEZE_W2V
    }
}, OUT_DIR/"w2v_trans_fix_model.pt")
print("✅ w2v_trans_fix 完成。")
