from typing import List
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def get_token_counts(sentences: List[str], tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
    """
    Return a list of token counts for each sentence.

    Args:
        sentences: A list of input sentences (strings).
        tokenizer: A pre-trained tokenizer (e.g., BERT, RoBERTa).

    Returns:
        token_counts: A list of integers, where each element represents the number of tokens for the corresponding sentence.
    """
    token_counts = [len(tokenizer.encode_plus(sentence, add_special_tokens=True)["input_ids"]) for sentence in sentences]
    return token_counts

def get_word_counts(sentences: List[str]):
    word_counts = [len(sentence.strip().split()) for sentence in sentences]
    return word_counts

def get_embeddings(input_texts, model: nn.Module, tokenizer, device):
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings