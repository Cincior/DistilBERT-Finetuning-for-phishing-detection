import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_NAME, SAVE_PATH, MAX_LENGTH, DEVICE
import os


def load_or_init_model():
    if os.path.exists(SAVE_PATH):
        print(f"Ładowanie modelu z {SAVE_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)
    else:
        print(f"Inicjalizacja nowego modelu: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    model.to(DEVICE)
    return tokenizer, model


def tokenize_dataset(df, tokenizer):
    def tokenize(example):
        return tokenizer(
            example["EmailText"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    ds = Dataset.from_pandas(
        df[["EmailText", "EmailLabel"]].rename(columns={"EmailLabel": "label"})
    )
    return ds.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def save_model(model, tokenizer):
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Model zapisany do {SAVE_PATH}")
