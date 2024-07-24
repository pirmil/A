import pandas as pd
import random
import numpy as np

def create_sample_dataframe(n_samples, id_qis, sentences_per_id_qi):
    # Ensure the inputs are valid
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(id_qis, list) or not id_qis:
        raise ValueError("id_qis must be a non-empty list.")
    if not isinstance(sentences_per_id_qi, dict) or not sentences_per_id_qi:
        raise ValueError("sentences_per_id_qi must be a non-empty dictionary.")
    if not all(k in sentences_per_id_qi for k in id_qis):
        raise ValueError("All id_qis must have corresponding sentences in sentences_per_id_qi.")
    sampled_id_qis = random.choices(id_qis, k=n_samples)
    sampled_sentences = [
        random.choice(sentences_per_id_qi[id_qi]) for id_qi in sampled_id_qis
    ]
    df = pd.DataFrame({
        'StoryBody': sampled_sentences,
        'id_qis': sampled_id_qis
    })
    return df

def select_first_n_occurences(df: pd.DataFrame, col='StoryBody', n=7):
    df["rpr"] = np.arange(len(df))
    result_df = df.groupby(col).head(n)
    return result_df.drop('rpr', axis=1), result_df["rpr"]

def main(seed=30):
    np.random.seed(seed)
    random.seed(seed)
    id_qis = ["id1", "id2", "id3"]
    sentences_per_id_qi = {
        "id1": ["This stock is overrated.", "The CEO is completely crazy.", "What a surprising deal."],
        "id2": ["I would like to buy their items.", "I love this firm.", "This is quite rare."],
        "id3": ["This has so much potential.", "I doubt they will make it.", "This new journey seems interesting."]
    }
    n_samples = 20
    df = create_sample_dataframe(n_samples, id_qis, sentences_per_id_qi)
    print(f"{df.shape[0]} rows")
    return df

def main_2_pos_2_neg(seed=30):
    np.random.seed(seed)
    random.seed(seed)
    id_qis = ["id1", "id2", "id3", "id5"]
    sentences_per_id_qi = {
        "id1": ["This stock is overrated.", "The CEO is completely crazy.", "This is a scandal."],
        "id2": ["I would like to buy their items.", "I love this firm.", "This is quite rare."],
        "id3": ["Excellent.", "This is interesting.", "Can't wait to see their new CEO!"],
        "id5": ["i dislike it so much.", "It deserves to be ignored."],
    }
    n_samples = 30
    df = create_sample_dataframe(n_samples, id_qis, sentences_per_id_qi)
    print(f"{df.shape[0]} rows")
    return df