"""
Preprocessing data for EDA and PEA
"""

import pandas as pd 
#### EDA

train_df = pd.read_csv("data/train.csv").rename(columns={"dialogue": "text"})


train_df["text_length"] = [len(x.split()) for x in train_df["text"]]
train_df["summary_length"] = [len(x.split()) for x in train_df["summary"]]
train_df["sum_text_ratio"] = train_df["summary_length"] / train_df["text_length"]
train_df.to_csv("visualizations/train_df.csv", index=False)

#### PEA

eval_df = pd.read_csv("visualizations/eval_df.csv")
eval_df["text_length"] = [len(x.split()) for x in eval_df["text"]]
eval_df["summary_length"] = [len(x.split()) for x in eval_df["summary"]]
eval_df["sum_text_ratio"] = eval_df["summary_length"] / eval_df["text_length"]

eval_df.to_csv("visualizations/eval_df.csv", index=False)
