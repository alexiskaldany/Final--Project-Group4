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
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate
import json
from torch.utils.data import Dataset
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
)
val = pd.read_csv(Path(data_dir / "validation.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)
test = pd.read_csv(Path(data_dir / "test.csv")).rename(
    columns={"dialogue": "text", "summary": "summary"}
)
output_dir = Path("output").absolute()
RUN_NAME = "blenderbot_small-90M"
run_path = Path(output_dir / RUN_NAME).absolute()
run_path.mkdir(parents=True, exist_ok=True)
MAX_LENGTH = 512
EPOCHS = 10
BATCH_SIZE = 8


print(output_dir)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    if torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
print(device)

def pre_process(row:pd.Series):
    inputs = tokenizer(row["text"], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    inputs["labels"] = tokenizer(row["summary"], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")["input_ids"]
    return inputs

class dialog_ds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.id = dataframe["id"]
        self.text = dataframe["text"].tolist()
        self.summary = dataframe["summary"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

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
        labels = self.tokenizer.encode_plus(
            self.summary[idx],
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
        }


tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M", ignore_mismatched_sizes=True)

train_dataset = dialog_ds(train, tokenizer, max_len=MAX_LENGTH)
val_dataset = dialog_ds(val, tokenizer, max_len=MAX_LENGTH)
test_dataset = dialog_ds(test, tokenizer, max_len=MAX_LENGTH)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)


rouge = evaluate.load("rouge", module_type="metric")


def get_metrics(evalPred: EvalPrediction):
    predictions = evalPred.predictions
    predictions = np.where(
        predictions != -100, predictions, tokenizer.pad_token_id
    ).astype(int)
    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    labels = evalPred.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(int)
    labels = tokenizer.batch_decode(
        sequences=labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    rouge.add_batch(references=labels, predictions=predictions)
    rouge_results = rouge.compute()
    return rouge_results


training_args = Seq2SeqTrainingArguments(
    overwrite_output_dir=True,
    learning_rate=3e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    lr_scheduler_type="linear",
    num_train_epochs=EPOCHS,
    seed=42,
    output_dir=run_path,
    run_name=RUN_NAME,
    predict_with_generate=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="steps",
    logging_steps=100,
    resume_from_checkpoint=True,
    save_strategy="epoch",
    disable_tqdm=True,
    
    
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=get_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
last_checkpoint = get_last_checkpoint(run_path)
if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()
results = trainer.evaluate()
trainer.save_metrics(split="all", metrics=results)
trainer.save_model(run_path)

