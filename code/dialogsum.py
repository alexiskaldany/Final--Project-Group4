""" 
diaglogsum.py
author: @alexiskaldany
created: 2022-11-22

Summarizing dialogues found in this dataset:
https://huggingface.co/datasets/knkarthick/dialogsum

DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 (Plus 100 holdout data for topic generation) dialogues with corresponding manually labeled summaries and topics.

Using "t5-base" model from huggingface's transformers library.
"""

### Loading data
from pathlib import Path
import pandas as pd
import os

data_dir = Path("data").absolute()
train = pd.read_csv(Path(data_dir / "train.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:100]
val = pd.read_csv(Path(data_dir / "validation.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:10]
test = pd.read_csv(Path(data_dir / "test.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)
output_dir = Path("output/test").absolute()
# print(output_dir)

### Configs
max_len = 1024
batch_size = 8

from torch.utils.data import Dataset


class dialog_ds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.id = dataframe["id"]
        self.text = dataframe["text"].tolist()
        self.summary = dataframe["summary"].tolist()
        self.text_tokenized = [
            tokenizer(
                self.text[i],
                padding=True,
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            for i in range(len(self.text))
        ]
        self.summary_tokenized = [
            tokenizer(
                self.summary[i],
                padding=True,
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            for i in range(len(self.summary))
        ]
        self.labels = [
            self.summary_tokenized[i]["input_ids"]
            for i in range(len(self.summary_tokenized))
        ]
        self.inputs = [
            self.text_tokenized[i]["input_ids"] for i in range(len(self.text_tokenized))
        ]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {
            "input_ids": (self.inputs[idx]),
            "labels": (self.labels[idx]),
            "attention_mask": (self.text_tokenized[idx]["attention_mask"]),
        }


from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
)
from torch.utils.data import DataLoader
import evaluate
import torch
import gc
from tqdm import tqdm

# Download configuration from huggingface.co and cache.
config = T5Config.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config)
tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=max_len)

rouge = evaluate.load('rouge')
train_ds = dialog_ds(train, tokenizer, max_len)
val_ds = dialog_ds(val, tokenizer, max_len)
test_ds = dialog_ds(test, tokenizer, max_len)

train_input = tokenizer(
    train["text"][0],
    padding=True,
    truncation=True,
    max_length=max_len,
    return_tensors="pt",
)
train_input["labels"] = tokenizer(
    train["summary"][0],
    padding=True,
    truncation=True,
    max_length=max_len,
    return_tensors="pt",
)["input_ids"]

# print(train_input.keys())
# print([x.shape for x in train_input.values()])
output = model(**train_input)
print(output.loss.item())
logits = output.logits
predicted_tokens = torch.argmax(logits, dim=-1)
print(predicted_tokens)
decoded = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
print(decoded)

metrics = rouge.compute(predictions=[decoded], references=[train["summary"][0]])
print(metrics)

# rouge = load_metric('rouge')
# rouge_test = rouge.compute(predictions=[decoded], references=[train['summary'][0]])
# print(rouge_test["rouge1"].mid.fmeasure)


class dialogTrainer:
    def __init__(self, model, tokenizer, max_len):
        """Loads model, tokenizer,max_len into the trainer"""
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eval_dfs = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def load_train_val_test(self, train: Dataset, val: Dataset, test: Dataset):
        """Loads train, val, and test data into the trainer"""
        self.train = train
        self.val = val
        self.test = test

    def load_training_args(self, lr, epochs, output_dir):
        """Loads training arguments into the trainer"""
        self.lr = lr
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.metric = evaluate.load('rouge')
        self.output_dir = output_dir
        self.number_of_steps = (len(self.train) + len(self.val)) * self.epochs
        self.tqdm_bar = tqdm(range(self.number_of_steps))

    def evaluate(self,epoch):
        """Evaluates the model on a dataset"""
        eval_df = pd.DataFrame(columns=["id","text", "summary"])
        summary = self.val.summary
        mode = "val"
        id = self.val.id
        loss = []
        pred_summary = []
        rouge_1 = []
        rouge_2 = []
        rouge_l = []
        for index, input in enumerate(self.val):
            self.tqdm_bar.update(1)
            input_ids = input["input_ids"].to(self.device)
            labels = input["labels"].to(self.device)
            attention_mask = input["attention_mask"].to(self.device)
            output = self.model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            loss.append(output.loss.item())
            logits = output.logits
            predicted_tokens = torch.argmax(logits, dim=-1)
            decoded = self.tokenizer.decode(
                predicted_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_summary.append(decoded)
            rouge = self.metric.compute(predictions=[decoded], references=[summary[index]])
            rouge_1.append(rouge["rouge1"])
            rouge_2.append(rouge["rouge2"])
            rouge_l.append(rouge["rougeL"])
        eval_df["id"] = id
        eval_df["mode"] = mode
        eval_df["text"] = self.val.text
        eval_df["summary"] = self.val.summary
        eval_df["pred_summary"] = pred_summary
        eval_df["rouge1"] = rouge_1
        eval_df["rouge2"] = rouge_2
        eval_df["rougeL"] = rouge_l
        eval_df["loss"] = loss
        eval_df["epoch"] = (epoch + 1)
        self.eval_dfs.append(eval_df)

    def training(self):
        """Trains the model"""
        self.train_dfs = []
        for epoch in range(self.epochs):
            mode = "train" 
            loss = []
            for index, input in enumerate(self.train):
                self.tqdm_bar.update(1)
                input_ids = input["input_ids"].to(self.device)
                labels = input["labels"].to(self.device)
                attention_mask = input["attention_mask"].to(self.device)
                output = self.model(
                    input_ids=input_ids, labels=labels, attention_mask=attention_mask
                )
                output.loss.backward()
                loss.append(output.loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_df = pd.DataFrame()
            train_df["id"] = self.train.id
            train_df["mode"] = mode
            train_df["text"] = self.train.text
            train_df["summary"] = self.train.summary
            train_df["loss"] = loss
            train_df["epoch"] = epoch + 1
            self.train_dfs.append(train_df)
            self.evaluate(epoch=epoch)
            
    def save_model(self):
        """Saves the model"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        concat_eval_df = pd.concat(self.eval_dfs)
        concat_eval_df.to_csv(os.path.join(self.output_dir, "eval_df.csv"))
        concat_train_df = pd.concat(self.train_dfs)
        concat_train_df.to_csv(os.path.join(self.output_dir, "train_df.csv"))


dtrainer = dialogTrainer(model, tokenizer, max_len)
dtrainer.load_train_val_test(train_ds, val_ds, test_ds)
dtrainer.load_training_args(lr=1e-5, epochs=25, output_dir=output_dir)
dtrainer.training()
dtrainer.save_model()
