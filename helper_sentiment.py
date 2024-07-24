import torch
import torch.nn as nn
from typing import List


def preprocess(text: str):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentiment(input_texts: List[str], model: nn.Module, tokenizer, device, as_softmax=True):
    input_texts = [preprocess(text) for text in input_texts]
    batch_dict = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**batch_dict).logits.cpu()
    if as_softmax:
        outputs = torch.softmax(outputs, dim=1)
    return outputs