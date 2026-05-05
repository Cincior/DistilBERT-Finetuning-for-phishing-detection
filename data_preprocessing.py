import re
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import SEED, TFIDF_MAX_FEATURES, COSINE_SIMILARITY_THRESHOLD, COSINE_BATCH_SIZE


def clean_email(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_and_clean_csv(filepath: str):
    dataset = load_dataset("csv", data_files=filepath)
    df = dataset["train"].to_pandas()

    df.rename(columns={"Email Text": "EmailText", "Email Type": "EmailLabel"}, inplace=True)
    df.dropna(subset=["EmailText", "EmailLabel"], inplace=True)
    df["EmailLabel"] = df["EmailLabel"].astype(int)
    df["EmailText"] = df["EmailText"].astype(str).str.strip()
    df["EmailText"] = df["EmailText"].apply(lambda x: re.sub(r"_+", " ", x))
    df["EmailText"] = df["EmailText"].apply(lambda x: re.sub(r"\s+", " ", x))
    df.drop_duplicates(subset=["EmailText"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def remove_near_duplicates(df):
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = vectorizer.fit_transform(df["EmailText"])

    num_rows = tfidf_matrix.shape[0]
    indices_to_drop = set()

    for i in range(0, num_rows, COSINE_BATCH_SIZE):
        end_i = min(i + COSINE_BATCH_SIZE, num_rows)
        chunk = tfidf_matrix[i:end_i]
        sim_matrix = cosine_similarity(chunk, tfidf_matrix)
        rows, cols = np.where(sim_matrix > COSINE_SIMILARITY_THRESHOLD)
        for r, c in zip(rows, cols):
            actual_r = r + i
            if actual_r < c:
                indices_to_drop.add(c)

    df.drop(index=list(indices_to_drop), inplace=True)
    df.dropna(subset=["EmailText", "EmailLabel"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def split_dataset(df):
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["EmailLabel"], random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df["EmailLabel"].iloc[temp_idx], random_state=SEED
    )
    df_train = df.loc[train_idx].copy().reset_index(drop=True)
    df_val = df.loc[val_idx].copy().reset_index(drop=True)
    df_test = df.loc[test_idx].copy().reset_index(drop=True)

    print("Po czyszczeniu:")
    print("Train:", df_train["EmailLabel"].value_counts().to_dict())
    print("Val:  ", df_val["EmailLabel"].value_counts().to_dict())
    print("Test: ", df_test["EmailLabel"].value_counts().to_dict())

    return df_train, df_val, df_test


def prepare_data(filepath: str):
    df = load_and_clean_csv(filepath)
    df = remove_near_duplicates(df)
    df["EmailText"] = df["EmailText"].apply(clean_email)
    return split_dataset(df)
