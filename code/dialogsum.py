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
)[:1000]
val = pd.read_csv(Path(data_dir / "validation.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:500]
test = pd.read_csv(Path(data_dir / "test.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)
output_dir = Path("output/test").absolute()
print(output_dir)

### Configs
max_len = 1024
batch_size = 8

from torch.utils.data import Dataset


class dialog_ds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
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
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_scheduler,
)
from torch.utils.data import DataLoader
from evaluate import evaluator
from datasets import load_metric
import torch
from tqdm import tqdm

# Download configuration from huggingface.co and cache.
config = T5Config.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config)
tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=max_len)
summarization_evaluator = evaluator("summarization")

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
# output = model(**train_input)
# print(output.loss)
# logits = output.logits
# predicted_tokens = torch.argmax(logits, dim=-1)
# print(predicted_tokens)
# decoded = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
# print(decoded)

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
        self.metric = load_metric("rouge")
        self.output_dir = output_dir
        self.number_of_steps = (len(self.train) + len(self.val)) * self.epochs
        self.tqdm_bar = tqdm(range(self.number_of_steps))

    def evaluate(self):
        """Evaluates the model on a dataset"""
        eval_df = pd.DataFrame(columns=["text", "summary", "pred_summary", "rouge1"])
        summary = self.val.summary
        pred_summary = []
        rouge_1 = []
        for index, input in enumerate(self.val):
            self.tqdm_bar.update(1)
            input_ids = input["input_ids"].to(self.device)
            labels = input["labels"].to(self.device)
            attention_mask = input["attention_mask"].to(self.device)
            output = self.model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            logits = output.logits
            predicted_tokens = torch.argmax(logits, dim=-1)
            decoded = self.tokenizer.decode(
                predicted_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_summary.append(decoded)
            rouge_1.append(
                self.metric.compute(predictions=[decoded], references=[summary[index]])[
                    "rouge1"
                ].mid.fmeasure
            )
        eval_df["text"] = self.val.text
        eval_df["summary"] = summary
        eval_df["pred_summary"] = pred_summary
        eval_df["rouge1"] = rouge_1
        self.eval_dfs.append(eval_df)

    def training(self):
        """Trains the model"""
        for epoch in range(self.epochs):
            for index, input in enumerate(self.train):
                self.tqdm_bar.update(1)
                input_ids = input["input_ids"].to(self.device)
                labels = input["labels"].to(self.device)
                attention_mask = input["attention_mask"].to(self.device)
                output = self.model(
                    input_ids=input_ids, labels=labels, attention_mask=attention_mask
                )
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.evaluate()

    def save_model(self):
        """Saves the model"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        concat_eval_df = pd.concat(self.eval_dfs)
        concat_eval_df.to_csv(os.path.join(self.output_dir, "eval_df.csv"))


dtrainer = dialogTrainer(model, tokenizer, max_len)
dtrainer.load_train_val_test(train_ds, val_ds, test_ds)
dtrainer.load_training_args(lr=1e-5, epochs=5, output_dir=output_dir)
dtrainer.training()
dtrainer.save_model()
