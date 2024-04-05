import os
import sys
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score ,precision_score, recall_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EvalPrediction,
)

input_file = sys.argv[1]
output_file = sys.argv[2]

if not input_file.endswith(".csv"):
  print('input_file has to be "csv" file')
  exit()

if not output_file.endswith(".csv"):
  print('output_file has to be "csv" file')
  exit()

class JTruthfulQADataset(Dataset):
    def __init__(self, X, y=None):
      self.X = X
      self.y = y

    def __len__(self):
      return len(self.X)

    def __getitem__(self, index):
      input = {
          "input_ids": [self.X[id]["input_ids"] for id in index],
          "attention_mask": [self.X[id]["attention_mask"] for id in index],
      }
      
      if self.y is not None:
        input["label"] = [self.y[id] for id in index]

      return input

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def RobertaEvaluation(input_file:str,output_file:str):

  cfg = {
    "model_name":"nlp-waseda/roberta_jtruthfulqa",
    "max_length":-1,
    "train_epoch":3,
    "lr":3e-5,
  }

  SEED = 42
  seed_everything(SEED)

  tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
  model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"])

  cfg["max_length"]=128

  def compute_metrics(p: EvalPrediction):
      preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
      preds = np.argmax(preds, axis=1)
      precision = precision_score(p.label_ids, preds)
      recall = recall_score(p.label_ids, preds)
      f1 = f1_score(p.label_ids, preds)
      accuracy = accuracy_score(p.label_ids, preds)
      return {"f1_score":f1,"precision":precision,"recall":recall,"accuracy":accuracy}

  trainer_args = TrainingArguments(
      seed=SEED,
      output_dir=".",
      overwrite_output_dir=False,
      do_train=True,
      do_eval=True,
      evaluation_strategy="epoch",
      logging_strategy="epoch",
      save_steps=1e6,
      log_level="critical",
      num_train_epochs=cfg["train_epoch"],
      learning_rate=cfg["lr"],
      per_device_train_batch_size=32,
      per_device_eval_batch_size=64,
      save_total_limit=1,
      fp16=False,
      remove_unused_columns=False,
      report_to="none"
  )

  # 推論用にデータを成形
  df = pd.read_csv('../data/JTruthfulQA.csv')
  q_list = df['Question'].tolist()
  y_preds = []
  test_df = pd.read_csv('../'+input_file)
  for i in range(len(df["generated"])):
    test_df["generated"][i] = q_list[i] + "　" + test_df["generated"]
  test_X = [tokenizer(text, padding="max_length", max_length=cfg["max_length"], truncation=True) for text in tqdm(test_df["generated"])]
  test_ds = JTruthfulQADataset(test_X)

  trainer = Trainer(
  model=model,
  args=trainer_args,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics,
  )

  # 推論
  test_preds = trainer.predict(test_ds)
  y_preds.append(np.argmax(test_preds.predictions, axis=1))

  # 評価結果の出力
  y_sub = sum(y_preds) / len(y_preds)
  y_sub = (y_sub > 0.5).astype(int)
  sub_df = pd.DataFrame({'question':q_list,'label':y_sub})
  sub_df.to_csv(output_file, index=False)

if __name__=="__main__":
    RobertaEvaluation(input_file,output_file)