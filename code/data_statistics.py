import pandas as pd 
import matplotlib.pyplot as plt

# train = pd.read_csv('data/train.csv')

# train["text_len"] = train["dialogue"].str.len()
# train["summary_len"] = train["summary"].str.len()

# train = train.drop(["dialogue", "summary","topic"], axis=1)
# train.to_csv("data/train_lengths.csv", index=False)

train = pd.read_csv("visualizations/train_lengths.csv")
fig, ax = plt.subplots(figsize=(10, 5))
"""Title of the histogram"""
ax.set_title('Text Lengths Histogram')
ax.set_xlabel('Text Length')
ax.set_ylabel('Count of Text Lengths')

ax.hist(train["text_len"], bins=100)

plt.savefig("visualizations/text_len_hist.png")

fig,ax = plt.subplots(figsize=(10,5))
ax.set_title('Summary Lengths Histogram')
ax.set_xlabel('Summary Length')
ax.set_ylabel('Count of Summary Lengths')

ax.hist(train["summary_len"], bins=100)
plt.savefig("visualizations/summary_len_hist.png")


describe =train.describe().round(0).astype(int)
describe.to_csv("visualizations/describe.csv")

