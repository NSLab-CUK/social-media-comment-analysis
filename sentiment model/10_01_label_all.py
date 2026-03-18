# -*- coding: utf-8 -*-要
"""
01_label_all.py
目标：对 DL_Cls_Out/train.raw.csv 全量贴情感标签（0负/1正），覆盖率=100%（按去重后行数）

输出：
  - DL_Cls_Out/train.seed.csv     
  - DL_Cls_Out/train.labeled.csv  （全量已标注）
可选：同时给 val/test 也打标签（把 LABEL_VAL_TEST = True 即可）
依赖：pandas numpy scikit-learn torch jieba
pip install pandas numpy scikit-learn torch jieba
"""

import re, math, random, pathlib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ===== 路径与参数 =====
OUT_DIR   = pathlib.Path("DL_Cls_Out"); OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_TRAIN = OUT_DIR / "train.raw.csv"
RAW_VAL_T = OUT_DIR / "val.text.csv"
RAW_TST_T = OUT_DIR / "test.text.csv"
ART       = OUT_DIR / "artifacts"; ART.mkdir(parents=True, exist_ok=True)

# 是否同时给 val/test 打标签（默认只标注 train）
LABEL_VAL_TEST = False

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 轻量训练参数（仅用于“辅助标注”的小模型）
BATCH=128; EPOCHS=4; LR=3e-4
MAX_VOCAB = 200_000
MAX_LEN   = 160
VAL_RATIO = 0.1
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# ===== 放宽的情感信号与口语模式 =====
emojis_pos = "😀😄😁😆😍😊👍❤💖🤩😎🎉👏🥳✨🙌🙂😺🫶🤝💯🌟😻😋😇"
emojis_neg = "😡😠🤬😢😭💔👎🙄😞😫😓😤🤯🥲😕☹️🪦🫠😖😣😩"
pat_pos = re.compile("[" + re.escape(emojis_pos) + "]")
pat_neg = re.compile("[" + re.escape(emojis_neg) + "]")
EMO_BRACKET = re.compile(r"\[[^\[\]\n]{1,8}\]")                 # [微笑][捂脸][允悲]…
LAUGH_RE = re.compile(r"(哈哈+|呵呵+|[hH]{2,}|233+|[xX]{2,})")  # 笑点
CRY_RE   = re.compile(r"(呜呜+|555+|T_T|QAQ|Orz|orz|>_<)")       # 哭点

# ✅ 颜文字正/负：使用 re.escape 安全转义，避免括号不平衡
EMOTICON_POS = re.compile("|".join(map(re.escape, [
    ":-)", ":)", "^_^", ":D", "(≧▽≦)/", "(^o^)/"
])))
EMOTICON_NEG = re.compile("|".join(map(re.escape, [
    ":-(", ":(", ">_<", ";-(", "T_T", "QAQ"
])))

EXCLA_RE = re.compile(r"[!！]{2,}")                              # 多叹号
QMARK_RE = re.compile(r"[?？]{2,}")                              # 多问号

lex_pos = {
    "好评","喜欢","满意","赞","优秀","推荐","友好","给力","真香","太棒","稳","利好","厉害",
    "惊艳","丝滑","顺利","成功","感谢","值了","值得","放心","口碑好","表现好","好用","强推","yyds",
    "绝绝子","爱了","上头","牛","顶","无敌","香","靠谱","好看","好听","惊喜","好评如潮","不错","可以","挺好","真好","开心"
}
lex_neg = {
    "差评","垃圾","失望","糟糕","崩了","坑人","踩雷","离谱","恶心","愤怒","吐了","破防","无语","背刺",
    "暴跌","跳水","闪崩","亏损","腰斩","跌停","割肉","套牢","崩溃","卡顿","卡死","宕机","bug",
    "问题多","投诉","维权","翻车","糟心","一塌糊涂","不行","太差","毁了","烂","烂透了","烂爆了",
    "服了","吐血","吐槽","黑心","智商税","无语子","烂大街","烂尾","辣鸡","稀烂","气死","栓Q",
    "一般","很一般","拉胯","一般般","无感","不推荐","别买","闹心","糟透了","太拉了","烦死了","惨","后悔"
}
negators = {"不","没","無","无","别","未","难","非","勿","并非","毫无","不是","不太","未必"}
contrast = {"但是","但","然而","不过","只是","却","可惜"}

def has_signal(s: str) -> bool:
    if not isinstance(s, str): s = str(s)
    return (
        bool(pat_pos.search(s) or pat_neg.search(s) or EMO_BRACKET.search(s)
             or EMOTICON_POS.search(s) or EMOTICON_NEG.search(s))
        or any(w in s for w in (lex_pos | lex_neg))
        or bool(LAUGH_RE.search(s) or CRY_RE.search(s) or EXCLA_RE.search(s) or QMARK_RE.search(s))
    )

def score_sent(s: str) -> int:
    # 正数偏正，负数偏负，0为无明显极性；仅用于做seed/覆盖，不是最终概率
    if not isinstance(s, str): s = str(s)
    s = s.strip()
    if len(s) < 3: return 0
    sc = 0
    # 表情/颜文字/笑哭模式
    if pat_pos.search(s) or EMO_BRACKET.search(s) or EMOTICON_POS.search(s) or LAUGH_RE.search(s): sc += 1
    if pat_neg.search(s) or EMOTICON_NEG.search(s) or CRY_RE.search(s): sc -= 2
    # 词典
    sc += sum(1 for w in lex_pos if w in s)
    sc -= sum(1 for w in lex_neg if w in s)
    # 多重叹问号
    if EXCLA_RE.search(s) and sc > 0: sc += 1
    if EXCLA_RE.search(s) and sc < 0: sc -= 1
    if QMARK_RE.search(s) and sc < 0: sc -= 1
    # 否定/转折折损
    if any(w in s for w in negators): sc = int(sc * 0.6)
    if any(w in s for w in contrast): sc = int(sc * 0.75)
    return sc

# ===== 分词 & 词表 =====
import jieba
def tokenize(text):
    text = re.sub(r"http\S+"," ", str(text))
    text = re.sub(r"@\S+"," ", text)
    return [t.strip() for t in jieba.cut(text) if t.strip()]

def build_vocab(texts, min_freq=3, max_size=MAX_VOCAB):
    c = Counter()
    for t in texts: c.update(tokenize(t))
    stoi = {"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
    for w,f in c.most_common():
        if f<min_freq or len(stoi)>=max_size: break
        stoi[w]=len(stoi)
    return stoi

class TextDS(Dataset):
    def __init__(self, df, vocab, max_len=MAX_LEN, has_label=True):
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist() if has_label and "label" in df.columns else None
        self.vocab=vocab; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def encode(self, toks):
        ids=[self.vocab.get(t,1) for t in toks]
        ids=[2]+ids[:self.max_len-2]+[3]
        att=[1]*len(ids)
        if len(ids)<self.max_len:
            pad=self.max_len-len(ids); ids += [0]*pad; att += [0]*pad
        return torch.tensor(ids), torch.tensor(att)
    def __getitem__(self, i):
        ids, att = self.encode(tokenize(self.texts[i]))
        if self.labels is None: return ids, att
        return ids, att, torch.tensor(self.labels[i])

# ===== 小型 Transformer（仅用于打标签，不是最终模型）=====
class PosEnc(nn.Module):
    def __init__(self, d=256, L=MAX_LEN):
        super().__init__()
        pe=torch.zeros(L,d); pos=torch.arange(0,L).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class TinyTrans(nn.Module):
    def __init__(self, V, emb=256, heads=4, layers=2, ff=1024, drop=0.2, pad=0):
        super().__init__()
        self.emb=nn.Embedding(V, emb, padding_idx=pad)
        self.pos=PosEnc(emb, MAX_LEN)
        enc=nn.TransformerEncoderLayer(d_model=emb, nhead=heads, dim_feedforward=ff,
                                       dropout=drop, batch_first=True)
        self.enc=nn.TransformerEncoder(enc, num_layers=layers)
        self.norm=nn.LayerNorm(emb); self.drop=nn.Dropout(drop)
        self.fc=nn.Linear(emb, 2)
    def forward(self, ids, att):
        x=self.pos(self.emb(ids)); mask=(att==0)
        x=self.enc(x, src_key_padding_mask=mask)
        x=(x*att.unsqueeze(-1)).sum(1)/(att.sum(1,keepdim=True)+1e-9)
        x=self.norm(x); return self.fc(self.drop(x))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__(); self.alpha=alpha; self.gamma=gamma
    def forward(self, logits, y):
        ce=nn.functional.cross_entropy(logits, y, weight=self.alpha, reduction="none")
        pt=torch.exp(-ce); loss=((1-pt)**self.gamma)*ce
        return loss.mean()

def train_small(df_train, df_val, vocab, save_to):
    ds_tr=TextDS(df_train, vocab); ds_va=TextDS(df_val, vocab)
    # 类均衡采样
    labels=[y for _,_,y in ds_tr]
    cnt=Counter(labels); w={c:1.0/max(n,1) for c,n in cnt.items()}
    sw=torch.tensor([w[y] for y in labels], dtype=torch.double)
    sampler=WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
    dl_tr=DataLoader(ds_tr, batch_size=BATCH, sampler=sampler)
    dl_va=DataLoader(ds_va, batch_size=BATCH)

    V=len(vocab)
    model=TinyTrans(V).to(DEVICE)
    cls=torch.tensor([cnt.get(0,1), cnt.get(1,1)], dtype=torch.float32, device=DEVICE)
    alpha=(cls.sum()/(2*cls)).clamp(max=5.0)
    crit=FocalLoss(alpha=alpha)
    opt=optim.AdamW(model.parameters(), lr=LR)

    best=0; patience=2; bad=0
    from sklearn.metrics import f1_score
    for ep in range(EPOCHS):
        model.train(); tot=0
        for ids,att,y in dl_tr:
            ids,att,y=ids.to(DEVICE),att.to(DEVICE),y.to(DEVICE)
            out=model(ids,att); loss=crit(out,y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tot+=loss.item()
        # val
        model.eval(); preds=[]; gold=[]
        with torch.no_grad():
            for ids,att,y in dl_va:
                ids,att,y=ids.to(DEVICE),att.to(DEVICE),y.to(DEVICE)
                p=model(ids,att).argmax(1)
                preds+=p.cpu().tolist(); gold+=y.cpu().tolist()
        f1=f1_score(gold,preds,average="macro")
        print(f"[bootstrap] ep{ep+1}/{EPOCHS} loss={tot/len(dl_tr):.4f} val_macroF1={f1:.4f}")
        if f1>best:
            best=f1; bad=0
            torch.save(model.state_dict(), save_to)
        else:
            bad+=1
            if bad>=patience:
                print("[early stop]"); break
    print(f"[save] 标注器 → {save_to} (val_macroF1={best:.4f})")
    return save_to

@torch.no_grad()
def infer_all(df, vocab, ckpt):
    ds=TextDS(df, vocab, has_label=False)
    dl=DataLoader(ds, batch_size=BATCH)
    model=TinyTrans(len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    probs=[]
    for ids,att in dl:
        ids,att=ids.to(DEVICE),att.to(DEVICE)
        p=torch.softmax(model(ids,att), dim=1)[:,1]
        probs+=p.cpu().tolist()
    return np.array(probs)

def read_text_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns:
        # 兜底：只有单列时当作 text
        if df.shape[1]==1:
            df.columns=["text"]
        else:
            raise ValueError(f"{path} 缺少 'text' 列")
    df = df[["text"]].dropna()
    df["text"] = df["text"].astype(str).str.replace(r"\s+"," ", regex=True).str.replace(r"\u200b","", regex=True).str.strip()
    return df.drop_duplicates("text").reset_index(drop=True)

def label_dataframe(df: pd.DataFrame, vocab, ckpt, seed: pd.DataFrame) -> pd.DataFrame:
    if len(df)==0: return df.assign(label=pd.Series(dtype=int))
    probs = infer_all(df, vocab, str(ckpt))
    model_labels = (probs >= 0.5).astype(int)
    out = df.copy(); out["label_model"] = model_labels
    # 用 seed 覆盖（seed 更干净）
    out = out.merge(seed, on="text", how="left", suffixes=("_model", ""))
    out["label"] = out["label"].where(out["label"].notna(), out["label_model"])
    out["label"] = out["label"].astype(int)
    return out[["text","label"]]

if __name__ == "__main__":
    # 读取 train.raw
    assert RAW_TRAIN.exists(), f"缺少 {RAW_TRAIN}，请先跑 Step1: 00_split_8_1_1.py"
    raw = read_text_csv(RAW_TRAIN)
    N = len(raw)
    print(f"[load] train.raw={N}")

    # 生成放宽规则 seed（强 + 中 + 有信号 + 可选长度兜底）
    tmp = raw.copy()
    tmp["score"] = tmp["text"].map(score_sent)
    high = tmp[tmp["score"].abs() >= 2]             # 强信号
    mid  = tmp[(tmp["score"].abs() == 1)]           # 中信号
    weak = tmp[(tmp["score"] == 0) & (tmp["text"].map(has_signal))]  # 有信号但强度不足

    # 可选“长度兜底”进一步扩大 seed（对较长句给弱极性）
    ENABLE_LEN_FALLBACK = True
    if ENABLE_LEN_FALLBACK:
        long_pos = tmp[(tmp["score"] == 0) &
                       (tmp["text"].str.len() >= 25) &
                       (tmp["text"].str.contains("真|太|很|超级|无敌|简直", regex=True) | tmp["text"].str.contains(EXCLA_RE))]
        long_pos = long_pos.assign(score=1)
        long_neg = tmp[(tmp["score"] == 0) &
                       (tmp["text"].str.len() >= 25) &
                       (tmp["text"].str.contains("服了|离谱|无语|烂|烦|气死|要命|崩溃|吐了", regex=True) | tmp["text"].str.contains(QMARK_RE))]
        long_neg = long_neg.assign(score=-1)
    else:
        long_pos = tmp.iloc[0:0]; long_neg = tmp.iloc[0:0]

    seed = pd.concat([high, mid, weak, long_pos, long_neg], ignore_index=True).drop_duplicates("text")
    seed["label"] = (seed["score"] > 0).astype(int)
    seed = seed[["text","label"]]
    seed.to_csv(OUT_DIR/"train.seed.csv", index=False, encoding="utf-8-sig")
    print(f"[seed] 种子：{len(seed)}（正={int(seed['label'].sum())} / 负={len(seed)-int(seed['label'].sum())}）")

    # 词表用 train.raw 全量构建（覆盖更全）
    vocab = build_vocab(raw["text"].tolist(), min_freq=3, max_size=MAX_VOCAB)

    # 训练小型标注器（仅用于辅助打标签）
    if len(seed) >= 50 and seed["label"].nunique() == 2:
        tr, va = train_test_split(seed, test_size=VAL_RATIO, random_state=SEED, stratify=seed["label"])
        ckpt = ART/"weak_annotator.ckpt"
        ckpt = train_small(tr, va, vocab, save_to=str(ckpt))
    else:
        # 极端情况下seed过少/单类：用极简规则直贴
        print("[warn] 种子过少/单类，退化为规则直贴（score>0→1，否则0）。")
        raw["label"] = (tmp["score"] > 0).astype(int)
        raw[["text","label"]].to_csv(OUT_DIR/"train.labeled.csv", index=False, encoding="utf-8-sig")
        print("✅ 已输出（规则直贴）：", OUT_DIR/"train.labeled.csv")
        raise SystemExit

    # 对 train.raw 全量推断 + 强规则覆盖 → 100% 覆盖
    labeled_train = label_dataframe(raw, vocab, ckpt, seed)
    assert len(labeled_train) == N and labeled_train["label"].notna().all()
    labeled_train.to_csv(OUT_DIR/"train.labeled.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 全量贴标完成：{len(labeled_train)}/{N} → {OUT_DIR/'train.labeled.csv'}")

    # （可选）同时对 val/test 打标签
    if LABEL_VAL_TEST:
        if RAW_VAL_T.exists():
            val_df = read_text_csv(RAW_VAL_T)
            labeled_val = label_dataframe(val_df, vocab, ckpt, seed)
            labeled_val.to_csv(OUT_DIR/"val.labeled.csv", index=False, encoding="utf-8-sig")
            print("→ 已输出：", OUT_DIR/"val.labeled.csv")
        if RAW_TST_T.exists():
            tst_df = read_text_csv(RAW_TST_T)
            labeled_tst = label_dataframe(tst_df, vocab, ckpt, seed)
            labeled_tst.to_csv(OUT_DIR/"test.labeled.csv", index=False, encoding="utf-8-sig")
            print("→ 已输出：", OUT_DIR/"test.labeled.csv")

    # 汇总分布
    cnt = Counter(labeled_train["label"])
    print(f"[stats] train.labeled 分布：正={cnt.get(1,0)}  负={cnt.get(0,0)}  覆盖率=100.0%")
