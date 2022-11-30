""" 
diaglogsum.py
author: @alexiskaldany
created: 2022-11-29

Summarizing dialogues found in this dataset:
https://huggingface.co/datasets/knkarthick/dialogsum

DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 (Plus 100 holdout data for topic generation) dialogues with corresponding manually labeled summaries and topics.

"""
from pathlib import Path
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load
import json
from torch.utils.data import Dataset
from datasets import load_metric
import torch
import gc
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")

###########
data_dir = Path("data").absolute()
train = pd.read_csv(Path(data_dir / "train.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:16]
val = pd.read_csv(Path(data_dir / "validation.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:16]
test = pd.read_csv(Path(data_dir / "test.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)[:16]
output_dir = Path("output").absolute()
RUN_NAME = "dialogsum_trainer_test"
MAX_LENGTH = 512
EPOCHS = 3

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    if torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
print(device)



class dialog_ds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.id = dataframe["id"]
        self.text = dataframe["text"].tolist()
        self.summary = dataframe["summary"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.text_tokenized = [
        #     tokenizer(
        #         self.text[i],
        #         padding=True,
        #         max_length=max_len,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     for i in range(len(self.text))
        # ]
        # self.summary_tokenized = [
        #     tokenizer(
        #         self.summary[i],
        #         padding=True,
        #         max_length=max_len,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     for i in range(len(self.summary))
        # ]
        # self.labels = [
        #     self.summary_tokenized[i]["input_ids"]
        #     for i in range(len(self.summary_tokenized))
        # ]
        # self.inputs = [
        #     self.text_tokenized[i]["input_ids"] for i in range(len(self.text_tokenized))
        # ]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.text[idx],
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self.tokenizer.encode_plus(self.summary[idx], max_length=self.max_len, truncation=True, padding=True, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "t5-small", ignore_mismatched_sizes=True
)
train_dataset = dialog_ds(train, tokenizer, max_len=MAX_LENGTH)
val_dataset = dialog_ds(val, tokenizer, max_len=MAX_LENGTH)
test_dataset = dialog_ds(test, tokenizer, max_len=MAX_LENGTH)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

perplexity = load("perplexity", module_type="metric")
training_args = Seq2SeqTrainingArguments(
    overwrite_output_dir=True,
    learning_rate=3e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    lr_scheduler_type="linear",
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=100,
    eval_steps=50,
    seed=42,
    metric_for_best_model="eval_perplexity",
    output_dir=output_dir,
    run_name=RUN_NAME,
    include_inputs_for_metrics = True,
)

blender_trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=perplexity,
    data_collator=data_collator,
)

blender_trainer.train()
results = blender_trainer.evaluate()
with open(output_dir / RUN_NAME / "results.json", "w") as f:
    json.dump(results, f)

blender_trainer.save_model(output_dir)
