# === Step 1.3: 用 teacher 给全体英文数据打标签，并做 8/1/1 切分 ===
import os, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

IN_CSV     = Path("PREP/prepared_corpus.csv")
TEACHER_PT = Path("Teacher_Cls/bert_teacher_model.pt")
OUT_DIR    = Path("DL_Cls_In"); OUT_DIR.mkdir(parents=True, exist_ok=True)

assert IN_CSV.exists(), f"找不到 {IN_CSV}"
assert TEACHER_PT.exists(), f"找不到 teacher 模型：{TEACHER_PT}"

df = pd.read_csv(IN_CSV)

text_col = "text_raw" if "text_raw" in df.columns else (
           "text_norm" if "text_norm" in df.columns else None)
assert text_col is not None, "需要 text_raw 或 text_norm 列"

if "quarter" not in df.columns:
    df["quarter"] = ""

print("总样本数：", len(df))

# 载入 teacher
ckpt = torch.load(TEACHER_PT, map_location="cpu")
MODEL_NAME = ckpt["model_name"]
labels     = ckpt["labels"]
label2id   = ckpt["label2id"]
id2label   = {i:c for c,i in label2id.items()}
num_classes = len(labels)

from transformers import AutoModel

class BertTeacher(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:,0]
        logits = self.fc(self.dropout(pooled))
        return logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = BertTeacher(MODEL_NAME, num_classes)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class UnlabeledDS(Dataset):
    def __init__(self, df, text_col, tokenizer, max_len=160):
        self.texts = df[text_col].astype(str).tolist()
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
        }

ds_all = UnlabeledDS(df, text_col, tokenizer, max_len=160)
dl_all = DataLoader(ds_all, batch_size=32, shuffle=False)

all_preds = []
with torch.no_grad():
    for batch in dl_all:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        pred_ids = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(pred_ids)

assert len(all_preds) == len(df)
df["label"] = [id2label[i] for i in all_preds]

print("打标完成，label 分布：")
print(df["label"].value_counts(normalize=True).round(3).to_dict())

# 8/1/1 分层切分
df_tmp, df_test = train_test_split(
    df, test_size=0.10, random_state=SEED, stratify=df["label"]
)
df_train, df_dev = train_test_split(
    df_tmp, test_size=0.111111, random_state=SEED, stratify=df_tmp["label"]
)

print("\n各子集 label 分布：")
for name, part in [("train", df_train), ("dev", df_dev), ("test", df_test)]:
    vc = part["label"].value_counts(normalize=True).round(3).to_dict()
    print(f"- {name}: {len(part)} | {vc}")

cols_out = ["doc_id"] if "doc_id" in df.columns else []
cols_out += [text_col, "label", "quarter"]

for name, part in [("train", df_train), ("dev", df_dev), ("test", df_test)]:
    part[cols_out].to_csv(OUT_DIR / f"{name}.csv",
                          index=False, encoding="utf-8-sig")

print("\n✅ 新的强标注数据已生成：", OUT_DIR.resolve(), " 下的 train/dev/test.csv")
print("现在你可以用这份数据重新跑 TF-IDF, TextCNN, BiLSTM, BERT 等所有后续步骤。")
