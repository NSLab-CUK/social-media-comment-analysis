# === Task3-B: Word2Vec + Transformer 头（train + val + test）===要m6
import os, re, numpy as np, pandas as pd, torch, torch.nn as nn, jieba
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from gensim.models import Word2Vec

TRAIN, VAL, TEST = "DL_Cls_Out/train.csv", "DL_Cls_Out/val.csv", "DL_Cls_Out/test.text.csv"
ART = "DL_Cls_Out/artifacts"; os.makedirs(ART, exist_ok=True)

MAX_LEN, EMB_DIM = 160, 300
BATCH_TRAIN, BATCH_EVAL = 128, 256
EPOCHS, LR = 8, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean(s): s=re.sub(r"\s+"," ",str(s)); s=re.sub(r"\u200b","",s); return s.strip()
def tok(s):  return [t for t in jieba.lcut(clean(s)) if t.strip()]

def read_tv():
    tr = pd.read_csv(TRAIN, encoding="utf-8-sig")[["text","label"]].dropna()
    va = pd.read_csv(VAL,   encoding="utf-8-sig")[["text","label"]].dropna()
    tr["text"]=tr["text"].map(clean); va["text"]=va["text"].map(clean)
    tr["label"]=tr["label"].astype(int); va["label"]=va["label"].astype(int)
    return tr, va
def read_test():
    te = pd.read_csv(TEST, encoding="utf-8-sig")
    if "text" not in te.columns: te.columns=["text"]
    te = te[["text"]].dropna(); te["text"]=te["text"].map(clean); return te

tr, va = read_tv(); te = read_test()

# 1) 训练 w2v
sentences = [tok(x) for x in pd.concat([tr["text"], va["text"]])]
w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=5, min_count=3, workers=4, sg=1, epochs=8)
w2v.save(os.path.join(ART,"w2v.model"))

# 2) 词表 & Embedding
stoi={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
for w in w2v.wv.index_to_key: stoi[w]=len(stoi)
emb=np.zeros((len(stoi),EMB_DIM),dtype="float32"); emb[1]=np.random.normal(0,0.1,EMB_DIM)
for w in w2v.wv.index_to_key: emb[stoi[w]]=w2v.wv[w]
emb=torch.tensor(emb)

def encode(text):
    ids=[stoi.get(t,1) for t in tok(text)]
    ids=[2]+ids[:MAX_LEN-2]+[3]
    att=[1]*len(ids)
    if len(ids)<MAX_LEN:
        pad=MAX_LEN-len(ids); ids+=[0]*pad; att+=[0]*pad
    return torch.tensor(ids), torch.tensor(att)

class DS(Dataset):
    def __init__(self, df, has_label=True):
        self.texts=df["text"].tolist(); self.labels=df["label"].astype(int).tolist() if has_label else None
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        ids,att=encode(self.texts[i])
        return (ids,att,torch.tensor(self.labels[i])) if self.labels is not None else (ids,att)

class TinyTransformer(nn.Module):
    def __init__(self, V, ff=1024, heads=4, layers=2, drop=0.3):
        super().__init__()
        self.emb=nn.Embedding(V, EMB_DIM, padding_idx=0); self.emb.weight.data.copy_(emb); self.emb.weight.requires_grad=True
        enc=nn.TransformerEncoderLayer(d_model=EMB_DIM, nhead=heads, dim_feedforward=ff, dropout=drop, batch_first=True)
        self.enc=nn.TransformerEncoder(enc, num_layers=layers)
        self.norm=nn.LayerNorm(EMB_DIM); self.drop=nn.Dropout(drop); self.fc=nn.Linear(EMB_DIM,2)
    def forward(self, ids, att):
        x=self.emb(ids); h=self.enc(x, src_key_padding_mask=(att==0))
        h=(h*att.unsqueeze(-1).float()).sum(1)/(att.sum(1,keepdim=True)+1e-9)
        return self.fc(self.norm(self.drop(h)))

tr_dl=DataLoader(DS(tr),batch_size=BATCH_TRAIN,shuffle=True)
va_dl=DataLoader(DS(va),batch_size=BATCH_EVAL)
te_dl=DataLoader(DS(te.assign(label=0),has_label=False),batch_size=BATCH_EVAL)

model=TinyTransformer(len(stoi)).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=LR)
best_f1, bad = -1, 0

for ep in range(1,EPOCHS+1):
    model.train(); run=0.0
    for ids,att,y in tr_dl:
        ids,att,y=ids.to(DEVICE),att.to(DEVICE),y.to(DEVICE)
        log=model(ids,att); loss=nn.functional.cross_entropy(log,y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); run+=loss.item()
    model.eval(); probs=[]; gold=[]
    with torch.no_grad():
        for ids,att,y in va_dl:
            ids,att,y=ids.to(DEVICE),att.to(DEVICE),y.to(DEVICE)
            p=torch.softmax(model(ids,att),dim=1)[:,1]; probs+=p.cpu().tolist(); gold+=y.cpu().tolist()
    probs=np.array(probs); gold=np.array(gold); pred=(probs>=0.5).astype(int)
    acc=accuracy_score(gold,pred); f1=f1_score(gold,pred,average="macro")
    print(f"[w2v+trans] ep{ep} val_acc={acc:.4f} val_macroF1={f1:.4f}")
    if f1>best_f1: best_f1, bad = f1, 0; torch.save(model.state_dict(), os.path.join(ART,"w2v_trans_best.ckpt"))
    else:
        bad+=1
        if bad>=3: print("[w2v+trans] early stop"); break

# 评估与测试预测
model.load_state_dict(torch.load(os.path.join(ART,"w2v_trans_best.ckpt"), map_location=DEVICE))
model.eval(); probs=[]; gold=[]
with torch.no_grad():
    for ids,att,y in va_dl:
        ids,att,y=ids.to(DEVICE),att.to(DEVICE),y.to(DEVICE)
        p=torch.softmax(model(ids,att),dim=1)[:,1]; probs+=p.cpu().tolist(); gold+=y.cpu().tolist()
probs=np.array(probs); gold=np.array(gold); pred=(probs>=0.5).astype(int)
print("\nVAL metrics:\n  acc=",round(accuracy_score(gold,pred),4)," macroF1=",round(f1_score(gold,pred,average='macro'),4))
print(classification_report(gold,pred,digits=3))

te_probs=[]
with torch.no_grad():
    for ids,att in te_dl:
        ids,att=ids.to(DEVICE),att.to(DEVICE)
        te_probs+=torch.softmax(model(ids,att),dim=1)[:,1].cpu().tolist()
te_probs=np.array(te_probs); te_pred=(te_probs>=0.5).astype(int)
out=te.copy(); out["prob_pos"]=te_probs; out["pred"]=te_pred
out_path="DL_Cls_Out/test.pred_w2v_trans.csv"; out.to_csv(out_path,index=False,encoding="utf-8-sig")
from collections import Counter
print("✅ TEST →", out_path); print("TEST 预测分布：", dict(Counter(te_pred)))
