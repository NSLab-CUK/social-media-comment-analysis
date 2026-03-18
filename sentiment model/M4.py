# === Task2 BERT编码 + Transformer头（训练+评估+测试预测） ===要m4
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# 路径 & 超参
MODEL   = "hfl/chinese-roberta-wwm-ext"
TRAIN_CSV = "DL_Cls_Out/train.csv"
VAL_CSV   = "DL_Cls_Out/val.csv"
TEST_TXT  = "DL_Cls_Out/test.text.csv"
SAVE_DIR  = "DL_Cls_Out/artifacts/task2_bert_topper/trans"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LEN = 160
BATCH_TRAIN = 16   # 训练时显存考虑
BATCH_EVAL  = 64
EPOCHS = 5
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True

# 数据工具
def _clean(df):
    df["text"] = (df["text"].astype(str)
                  .str.replace(r"\s+"," ",regex=True)
                  .str.replace(r"\u200b","",regex=True)
                  .str.strip())
    return df

def load_train_val():
    tr = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")[["text","label"]].dropna()
    va = pd.read_csv(VAL_CSV,   encoding="utf-8-sig")[["text","label"]].dropna()
    tr, va = _clean(tr), _clean(va)
    tr["label"]=tr["label"].astype(int); va["label"]=va["label"].astype(int)
    return tr, va

def load_test():
    te = pd.read_csv(TEST_TXT, encoding="utf-8-sig")
    if "text" not in te.columns:
        if te.shape[1]==1: te.columns=["text"]
        else: raise ValueError("test.text.csv 缺少 text 列")
    te = _clean(te[["text"]].dropna())
    return te

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len=160):
        self.texts=texts; self.labels=labels; self.tok=tok; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc=self.tok(self.texts[i], truncation=True, padding="max_length",
                     max_length=self.max_len, return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        if self.labels is not None:
            item["labels"]=torch.tensor(int(self.labels[i]),dtype=torch.long)
        return item

# Transformer 头
class TinyTransformerHead(nn.Module):
    def __init__(self, d_model=768, heads=4, layers=2, ff=1024, drop=0.2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm= nn.LayerNorm(d_model); self.drop=nn.Dropout(drop); self.fc=nn.Linear(d_model, 2)
    def forward(self, x, mask):
        key_padding=(mask==0)
        h = self.enc(x, src_key_padding_mask=key_padding)
        mask=mask.unsqueeze(-1).float()
        h=(h*mask).sum(1)/(mask.sum(1)+1e-9)
        return self.fc(self.norm(h))

# 1) 载入 tokenizer、冻结 BERT
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
bert = AutoModel.from_pretrained(MODEL).to(DEVICE)
for p in bert.parameters(): p.requires_grad=False
bert.eval()

# 2) 数据
tr_df, va_df = load_train_val()
te_df = load_test()
tr_dl = DataLoader(DS(tr_df["text"].tolist(), tr_df["label"].tolist(), tok, MAX_LEN), batch_size=BATCH_TRAIN, shuffle=True)
va_dl = DataLoader(DS(va_df["text"].tolist(), va_df["label"].tolist(), tok, MAX_LEN), batch_size=BATCH_EVAL)
te_dl = DataLoader(DS(te_df["text"].tolist(), None, tok, MAX_LEN), batch_size=BATCH_EVAL)

# 3) 训练
head = TinyTransformerHead().to(DEVICE)
opt  = torch.optim.AdamW(head.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=FP16 and torch.cuda.is_available())

best_f1, bad = -1, 0
for ep in range(1, EPOCHS+1):
    head.train(); run=0.0
    for i,b in enumerate(tr_dl, start=1):
        ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["labels"].to(DEVICE)
        with torch.no_grad():
            seq = bert(input_ids=ids, attention_mask=att).last_hidden_state
        with torch.cuda.amp.autocast(enabled=FP16):
            log = head(seq, att)
            loss = nn.functional.cross_entropy(log, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        run += loss.item()
        if i%50==0: print(f"[trans] ep{ep} step{i}/{len(tr_dl)} loss={run/i:.4f}")

    # 验证
    head.eval(); probs=[]; gold=[]
    with torch.no_grad():
        for b in va_dl:
            ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["labels"].to(DEVICE)
            seq=bert(input_ids=ids, attention_mask=att).last_hidden_state
            p=torch.softmax(head(seq, att), dim=1)[:,1]
            probs+=p.cpu().tolist(); gold+=y.cpu().tolist()
    probs=np.array(probs); gold=np.array(gold); pred=(probs>=0.5).astype(int)
    acc=accuracy_score(gold,pred); f1=f1_score(gold,pred,average="macro")
    print(f"[trans] ep{ep} VAL acc={acc:.4f} macroF1={f1:.4f}")
    if f1>best_f1:
        best_f1=f1; bad=0
        torch.save(head.state_dict(), os.path.join(SAVE_DIR,"best.ckpt"))
        print(f"✅ 保存最优 → {os.path.join(SAVE_DIR,'best.ckpt')}")
    else:
        bad+=1
        if bad>=3:
            print("[trans] early stop"); break

# 4) 验证报告 + 测试集预测
head.load_state_dict(torch.load(os.path.join(SAVE_DIR,"best.ckpt"), map_location=DEVICE))
head.eval()
probs=[]; gold=[]
with torch.no_grad():
    for b in va_dl:
        ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["labels"].to(DEVICE)
        seq=bert(input_ids=ids, attention_mask=att).last_hidden_state
        p=torch.softmax(head(seq, att), dim=1)[:,1]
        probs+=p.cpu().tolist(); gold+=y.cpu().tolist()
probs=np.array(probs); gold=np.array(gold); pred=(probs>=0.5).astype(int)
print("\nVAL metrics:")
print("  acc =", round(accuracy_score(gold,pred),4), " macroF1 =", round(f1_score(gold,pred,average='macro'),4))
print(classification_report(gold,pred,digits=3))

te_probs=[]
with torch.no_grad():
    for b in te_dl:
        ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE)
        seq=bert(input_ids=ids, attention_mask=att).last_hidden_state
        te_probs+=torch.softmax(head(seq, att), dim=1)[:,1].cpu().tolist()
te_probs=np.array(te_probs); te_pred=(te_probs>=0.5).astype(int)
out=te_df.copy(); out["prob_pos"]=te_probs; out["pred"]=te_pred
out_path="DL_Cls_Out/test.pred_task2_trans.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")
from collections import Counter
print("✅ TEST 推理完成 →", out_path)
print("TEST 预测分布：", dict(Counter(te_pred)))
