"""
Preprocessing data for EDA and PEA
"""

import pandas as pd 

#### EDA

train_df = pd.read_csv("data/train.csv")


train_df["text_length"] = train_df["dialogue"].str.len()
train_df["summary_length"] = train_df["summary"].str.len()
train_df["sum_text_ratio"] = train_df["summary_length"] / train_df["text_length"]
train_df.to_csv("visualizations/train_df.csv", index=False)

#### PEA

eval_df = pd.read_csv("visualizations/eval_df.csv")
eval_df["text_length"] = eval_df["text"].str.len()
eval_df["summary_length"] = eval_df["summary"].str.len()
eval_df["sum_text_ratio"] = eval_df["summary_length"] / eval_df["text_length"]

eval_df.to_csv("visualizations/eval_df.csv", index=False)
