from typing import List
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)

def preprocess(text: str):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentiment(input_texts: List[str], model: nn.Module=model, tokenizer=tokenizer, device=device, as_softmax=True):
    input_texts = [preprocess(text) for text in input_texts]
    batch_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**batch_dict).logits.cpu()
    if as_softmax:
        outputs = torch.softmax(outputs, dim=1)
    return outputs


import pandas as pd
import numpy as np
from helper_sentiment import get_sentiment

t_TSLA = ['Tesla is doing great!', 'I like Tesla', "Elon Musk is the best CEO ever", "Tesla is average.", "This shows that they are not as good as they used to be."]
t_MSFT = ['I hate microsoft', 'Microsoft is quite good', 'I do not know how Microsoft still exists']

ds = ['2018-01-01', '2018-01-03']

df = pd.DataFrame(
    {"date": [ds[0], ds[0], ds[0], ds[0], ds[1], ds[1], ds[1]],
     "id_qis": ['TSLA', 'MSFT', 'MSFT', 'TSLA', 'TSLA', 'TSLA', 'MSFT'],
     "Body": [t_TSLA[0], t_MSFT[0], t_MSFT[1], t_TSLA[1], t_TSLA[2], t_TSLA[3], t_MSFT[2]]}
)
display(df)
sent = pd.DataFrame(get_sentiment(df['Body'].to_list(), as_softmax=True), index=df.index, columns=['Negative', 'Neutral', 'Positive'])
df = pd.concat((df, sent), axis=1)
df = df.drop('Body', axis=1).groupby(['date', 'id_qis']).mean().reset_index(drop=False)
df['Sent'] = df['Positive'] - df['Negative']
df.drop(["Negative", "Neutral", "Positive"], axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'])
display(df)
def func(x):
    print(x)

def create_rolling(df: pd.DataFrame, windows):
    df.set_index('date', inplace=True)
    for w in windows:
        df[f'Sent_{w}'] = df['Sent'].rolling(w, min_periods=1).mean()
    return df

df.groupby('id_qis').apply(create_rolling, windows=['7D', '30D'], include_groups=False).swaplevel(i='id_qis', j='date').sort_index()
# df.groupby('id_qis').apply(func, include_groups=False)
