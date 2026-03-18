# === Task2 推理一键版：BERT编码 + LSTM头 ===要m3
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

MODEL   = "hfl/chinese-roberta-wwm-ext"
VAL_CSV = "DL_Cls_Out/val.csv"
TEST_TXT= "DL_Cls_Out/test.text.csv"
CKPT    = "DL_Cls_Out/artifacts/task2_bert_topper/lstm/best.ckpt"  # 你上面训练产物
MAX_LEN = 160
BATCH   = 64
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# --- 头部结构（与训练一致） ---
class LSTMHead(nn.Module):
    def __init__(self, d_model=768, hid=256, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_model, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop); self.fc = nn.Linear(hid*2, 2)
    def forward(self, x, mask):
        out,_ = self.lstm(x)
        mask = mask.unsqueeze(-1).float()
        out = (out*mask).sum(1) / (mask.sum(1)+1e-9)
        return self.fc(self.drop(out))

# --- 数据集 ---
class DSVal(Dataset):
    def __init__(self, df, tok):
        self.texts=df["text"].astype(str).tolist()
        self.labels=df["label"].astype(int).tolist()
        self.tok=tok
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        item["labels"]=torch.tensor(self.labels[i], dtype=torch.long)
        return item

class DSTest(Dataset):
    def __init__(self, df, tok):
        self.texts=df["text"].astype(str).tolist(); self.tok=tok
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        return {k:v.squeeze(0) for k,v in enc.items()}

# --- 读取数据 ---
def _load_csvs():
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")[["text","label"]].dropna()
    val_df["text"]=val_df["text"].astype(str).str.replace(r"\s+"," ",regex=True).str.replace(r"\u200b","",regex=True).str.strip()
    test_df= pd.read_csv(TEST_TXT, encoding="utf-8-sig")
    if "text" not in test_df.columns:
        if test_df.shape[1]==1: test_df.columns=["text"]
        else: raise ValueError("test.text.csv 缺少 text 列")
    test_df=test_df[["text"]].dropna()
    test_df["text"]=test_df["text"].astype(str).str.replace(r"\s+"," ",regex=True).str.replace(r"\u200b","",regex=True).str.strip()
    return val_df, test_df

# --- 主流程 ---
tok  = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
bert = AutoModel.from_pretrained(MODEL).to(DEVICE)
for p in bert.parameters(): p.requires_grad=False
bert.eval()

assert os.path.exists(CKPT), f"找不到权重：{CKPT}"
head = LSTMHead().to(DEVICE)
head.load_state_dict(torch.load(CKPT, map_location=DEVICE))
head.eval()

val_df, test_df = _load_csvs()

# 验证集评估
va_dl = DataLoader(DSVal(val_df, tok), batch_size=BATCH)
probs, gold = [], []
with torch.no_grad():
    for b in va_dl:
        ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE); y=b["labels"].to(DEVICE)
        seq = bert(input_ids=ids, attention_mask=att).last_hidden_state
        log = head(seq, att)
        p = torch.softmax(log, dim=1)[:,1]
        probs += p.cpu().tolist(); gold += y.cpu().tolist()
probs=np.array(probs); gold=np.array(gold); pred=(probs>=0.5).astype(int)
print("VAL metrics:")
print("  acc =", round(accuracy_score(gold, pred),4), " macroF1 =", round(f1_score(gold, pred, average='macro'),4))
print(classification_report(gold, pred, digits=3))

# 测试集预测
te_dl = DataLoader(DSTest(test_df, tok), batch_size=BATCH)
te_probs=[]
with torch.no_grad():
    for b in te_dl:
        ids=b["input_ids"].to(DEVICE); att=b["attention_mask"].to(DEVICE)
        seq = bert(input_ids=ids, attention_mask=att).last_hidden_state
        log = head(seq, att)
        te_probs += torch.softmax(log, dim=1)[:,1].cpu().tolist()
te_probs=np.array(te_probs); te_pred=(te_probs>=0.5).astype(int)
out = test_df.copy(); out["prob_pos"]=te_probs; out["pred"]=te_pred
out_path = "DL_Cls_Out/test.pred_task2_lstm.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")
from collections import Counter
print("✅ TEST 推理完成 →", out_path)
print("TEST 预测分布：", dict(Counter(te_pred)))
