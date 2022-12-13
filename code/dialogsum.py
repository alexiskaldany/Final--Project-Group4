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
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
)
from torch.utils.data import DataLoader, Dataset
import evaluate
import torch
import gc
from tqdm import tqdm

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
current_run = "with_summarize"
previous_run = ""
previous_checkpoint = output_dir / previous_run

###########
### Configs
NUM_EPOCHS = 2
MAX_LEN = 512
LOAD_FROM_CHECKPOINT = False
config = T5Config.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config)
tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=MAX_LEN)
rouge = evaluate.load("rouge")
###########


class dialog_ds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.id = dataframe["id"]
        self.text = dataframe["text"].tolist()
        self.summary = dataframe["summary"].tolist()
        self.text_tokenized = [
            tokenizer(
            ("summarize:" + self.text[i]),
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


class dialogTrainer:
    def __init__(self, model, tokenizer, max_len):
        """
        Loads model, tokenizer,max_len into the trainer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eval_dfs = []
        self.eval_average_dfs = []
        self.train_dfs = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def load_train_val_test(self, train: Dataset, val: Dataset, test: Dataset):
        """
        Loads train, val, and test data into the trainer
        """
        self.train = train
        self.val = val
        self.test = test

    def load_training_args(
        self, epochs, output_dir, current_run, previous_run: str = ""
    ):
        """
        Loads training arguments into the trainer
        epochs: Number of epochs to train
        output_dir: Directory to save model checkpoints
        optimizer: Optimizer to use for training
        current_run: Current run number
        previous_run: Previous run number
        """
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.metric = evaluate.load("rouge")
        self.output_dir = output_dir
        self.current_run = current_run
        self.current_output_path = self.output_dir / self.current_run
        self.previous_run = previous_run
        self.previous_output_path = self.output_dir / self.previous_run
        if not self.current_output_path.exists():
            self.current_output_path.mkdir(parents=True)
        self.number_of_steps = (len(self.train) + len(self.val)) * self.epochs + len(
            self.test
        )
        self.tqdm_bar = tqdm(range(self.number_of_steps))
        self.tqdm_bar.set_description(
            f"Using {self.device}"
        )

    def evaluate(self, epoch):
        """
        Evaluates the model on a dataset
        """
        eval_df = pd.DataFrame(columns=["id", "text", "summary"])
        summary = self.val.summary
        self.mode = "val"
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
            rouge = self.metric.compute(
                predictions=[decoded], references=[summary[index]]
            )
            rouge_1.append(rouge["rouge1"])
            rouge_2.append(rouge["rouge2"])
            rouge_l.append(rouge["rougeL"])
        eval_df["id"] = id
        eval_df["mode"] = self.mode
        eval_df["text"] = self.val.text
        eval_df["summary"] = self.val.summary
        eval_df["pred_summary"] = pred_summary
        eval_df["rouge1"] = rouge_1
        eval_df["rouge2"] = rouge_2
        eval_df["rougeL"] = rouge_l
        eval_df["loss"] = loss
        eval_df["epoch"] = epoch + 1
        average = pd.DataFrame(
            eval_df[["loss", "rouge1", "rouge2", "rougeL"]].mean()
        ).T.round(4)
        self.eval_dfs.append(eval_df.round(4))
        self.eval_average_dfs.append(average)

    def training(self):
        """
        Trains the model
        Also triggers the evaluation, test, and model saving functions
        """
        # self.train_dfs = []

        for epoch in range(self.epochs):
            loss_list = []
            rouge1_list = []
            self.mode = "train"
            for index, input in enumerate(self.train):
                self.tqdm_bar.update(1)
                # .set_postfix(
                #     f"Epoch {epoch} {self.mode} {index}"
                # )
                input_ids = input["input_ids"].to(self.device)
                labels = input["labels"].to(self.device)
                attention_mask = input["attention_mask"].to(self.device)
                output = self.model(
                    input_ids=input_ids, labels=labels, attention_mask=attention_mask
                )
                output.loss.backward()
                predicted_tokens = torch.argmax(output.logits, dim=-1)
                decoded = self.tokenizer.decode(
                    predicted_tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                rouge1_list.append(
                    self.metric.compute(
                        predictions=[decoded], references=[self.train.summary[index]]
                    )["rouge1"]
                )
                loss_list.append(output.loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
            df = pd.DataFrame(columns=["index", "epoch", "rouge1", "loss"])
            df["index"] = range(len(loss_list))
            df["epoch"] = epoch + 1
            df["rouge1"] = rouge1_list
            df["loss"] = loss_list
            self.train_dfs.append(df.round(4))
            self.evaluate(epoch=epoch)
            self.save_model()
        self.run_test()
        self.tqdm_bar.close()
        self.save_model()

    def run_test(self):
        """
        A test loop to generate statistics to analyze the performace of the model
        """
        self.mode = "test"
        self.output_list = []
        for index, input in enumerate(self.test):
            row_dict = {
                "summary": self.test.summary[index],
            }
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
            row_dict["pred_summary"] = decoded
            rouge = self.metric.compute(
                predictions=[decoded], references=[self.test.summary[index]]
            )
            row_dict["rouge1"] = rouge["rouge1"]
            row_dict["rouge2"] = rouge["rouge2"]
            row_dict["rougeL"] = rouge["rougeL"]
            row_dict["loss"] = output.loss.item()
            self.output_list.append(row_dict)
        self.output_df = pd.DataFrame(self.output_list).round(4)
        self.averages_df = pd.DataFrame(
            self.output_df[["rouge1", "rouge2", "rougeL", "loss"]].mean()
        ).T.round(4)
        eval_averages_df = pd.concat(self.eval_average_dfs, axis=0).round(4)
        eval_averages_df.to_csv(
            self.current_output_path / "eval_averages.csv", index=False
        )
        self.output_df.to_csv(self.current_output_path / "full_test_output.csv")
        self.averages_df.to_csv(self.current_output_path / "averages_test_output.csv")

    def save_model(self):
        """
        Saves the model inside the current_output_path directory
        """
        train_dfs = pd.concat(self.train_dfs, axis=0).round(4)
        train_dfs.to_csv(self.current_output_path / "train_dfs.csv", index=False)
        self.model.save_pretrained(self.current_output_path)
        self.tokenizer.save_pretrained(self.current_output_path)
        concat_eval_df = pd.concat(self.eval_dfs).round(4)
        concat_eval_df.to_csv(self.current_output_path / "eval_df.csv")

    def load_checkpoint(self, checkpoint_path):
        """
        Loads the model from a checkpoint
        checkpoint_path: the directory where the previous model was stored
        """
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
        self.model.to(self.device)


##############
train_ds = dialog_ds(train, tokenizer, MAX_LEN)
val_ds = dialog_ds(val, tokenizer, MAX_LEN)
test_ds = dialog_ds(test, tokenizer, MAX_LEN)

dtrainer = dialogTrainer(model, tokenizer, MAX_LEN)
if LOAD_FROM_CHECKPOINT:
    dtrainer.load_checkpoint(checkpoint_path=previous_checkpoint)
dtrainer.load_train_val_test(train_ds, val_ds, test_ds)
dtrainer.load_training_args(
    epochs=NUM_EPOCHS,
    output_dir=output_dir,
    current_run=current_run,
    previous_run=previous_run,
)
dtrainer.training()


##########################
# train_input = tokenizer(
#     train["text"][0],
#     padding=True,
#     truncation=True,
#     max_length=max_len,
#     return_tensors="pt",
# )
# train_input["labels"] = tokenizer(
#     train["summary"][0],
#     padding=True,
#     truncation=True,
#     max_length=max_len,
#     return_tensors="pt",
# )["input_ids"]

# print(train_input.keys())
# print([x.shape for x in train_input.values()])
# output = model(**train_input)
# print(output.loss.item())
# logits = output.logits
# predicted_tokens = torch.argmax(logits, dim=-1)
# print(predicted_tokens)
# decoded = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
# print(decoded)

# metrics = rouge.compute(predictions=[decoded], references=[train["summary"][0]])
# print(metrics)

# rouge = load_metric('rouge')
# rouge_test = rouge.compute(predictions=[decoded], references=[train['summary'][0]])
# print(rouge_test["rouge1"].mid.fmeasure)
