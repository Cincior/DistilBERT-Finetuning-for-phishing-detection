import seaborn as sns
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TrainingMetricsCallback import *

model_name = "distilbert-base-uncased"
save_path = "./saved_modelT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }


def clean_email(text):
    text = text.lower()
    text = re.sub(r'\b(http|https|www)\S+', 'httpurl', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_dataset(df_a):
    def tokenize(example):
        return tokenizer(
            example["EmailText"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    ds = Dataset.from_pandas(df_a[["EmailText", "EmailLabel"]].rename(columns={"EmailLabel": "label"}))
    return ds.map(tokenize, batched=True)


if os.path.exists(save_path):
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("csv", data_files="phishingEmail.csv")

    df = dataset["train"].to_pandas()
    df.rename(columns={"Email Text": "EmailText", "Email Type": "EmailLabel"}, inplace=True)
    df.dropna(subset=["EmailText", "EmailLabel"], inplace=True)
    df["EmailLabel"] = df["EmailLabel"].astype(int)
    df["EmailText"] = df["EmailText"].astype(str).str.strip()
    df["EmailText"] = df["EmailText"].apply(lambda x: re.sub(r"_+", " ", x))
    df["EmailText"] = df["EmailText"].apply(lambda x: re.sub(r"\s+", " ", x))
    df.drop_duplicates(subset=["EmailText"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["EmailText"])

    batch_size = 2000
    num_rows = tfidf_matrix.shape[0]
    indices_to_drop = set()

    for i in range(0, num_rows, batch_size):
        end_i = min(i + batch_size, num_rows)
        chunk = tfidf_matrix[i:end_i]
        sim_matrix = cosine_similarity(chunk, tfidf_matrix)
        rows, cols = np.where(sim_matrix > 0.9)
        for r, c in zip(rows, cols):
            actual_r = r + i
            if actual_r < c:
                indices_to_drop.add(c)

    df.drop(index=list(indices_to_drop), inplace=True)
    df.dropna(subset=["EmailText", "EmailLabel"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_idx, temp_idx = sklearn_train_test_split(
        df.index,
        test_size=0.2,
        stratify=df["EmailLabel"],
        random_state=SEED
    )
    val_idx, test_idx = sklearn_train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=df["EmailLabel"].iloc[temp_idx],
        random_state=SEED
    )

    df_train = df.loc[train_idx].copy().reset_index(drop=True)
    df_val = df.loc[val_idx].copy().reset_index(drop=True)
    df_test = df.loc[test_idx].copy().reset_index(drop=True)

    df_train["EmailText"] = df_train["EmailText"].apply(clean_email)
    df_val["EmailText"] = df_val["EmailText"].apply(clean_email)
    df_test["EmailText"] = df_test["EmailText"].apply(clean_email)

    print("Po czyszczeniu:")
    print("Train:", df_train["EmailLabel"].value_counts().to_dict())
    print("Val:", df_val["EmailLabel"].value_counts().to_dict())
    print("Test:", df_test["EmailLabel"].value_counts().to_dict())

    train_ds = tokenize_dataset(df_train)
    val_ds = tokenize_dataset(df_val)
    test_ds = tokenize_dataset(df_test)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED
    )

    metrics_callback = TrainingMetricsCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )

    trainer.train()

    print("\n=== FINAL TEST METRICS ===")
    test_results = trainer.evaluate(test_ds)
    for key, value in test_results.items():
        print(f"{key}: {round(value, 4)}")

    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    print("\n=== CONFUSION MATRIX (TEST SET) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Phishing'],
                yticklabels=['Normal', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    plot_all_metrics(metrics_callback)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.to(device)


def predict_proba(texts):
    all_probs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            stride=64,
            return_overflowing_tokens=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "overflow_to_sample_mapping"}
        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        avg_probs = np.mean(probs, axis=0)
        all_probs.append(avg_probs)
    return np.array(all_probs)


def get_suspicious_fragments(text, lime_words):
    words = text.split()
    fragments = []
    for lw, score in lime_words:
        if score > 0:
            for i, w in enumerate(words):
                if lw.lower() in w.lower():
                    start = max(0, i - 2)
                    end = min(len(words), i + 3)
                    fragment = " ".join(words[start:end])
                    fragments.append((fragment, score))
    return fragments


def predict_proba_raw(texts):
    cleaned_texts = [clean_email(t) for t in texts]
    return predict_proba(cleaned_texts)


explainer = LimeTextExplainer(class_names=["normal", "phishing"])


def explain_prediction(text):
    probs = predict_proba_raw([text])[0]
    print("\n" + "=" * 30)
    print("TEXT:", text[:100], "...")
    print(f"Normal: {round(probs[0] * 100, 2)}% | Phishing: {round(probs[1] * 100, 2)}%")

    exp = explainer.explain_instance(
        text,
        predict_proba_raw,
        num_features=5,
        num_samples=500
    )

    suspicious_words = [(word, score) for word, score in exp.as_list() if score > 0]
    print("\nSuspicious words:")
    if not suspicious_words:
        print("None found.")
    else:
        for word, score in suspicious_words:
            print(f"[ {word} ] -> {round(score, 3)}")


explain_prediction(
    "Dear Customer enron,    We detected an unusual sign-in attempt on your account from a new device and "
    "location. For your security, your account has been temporarily limited. Please confirm your "
    "identity and restore full access by clicking the secure link below: Verify My Account Now → "
    "http://security-check-account.com/verify If you do not verify your account within 24 hours, "
    "your account will be permanently suspended. Thank you for your cooperation, Security Team")
explain_prediction(
    "URGENT: Your PayPal account has been suspended. Please log in at http://secure-paypal-login.com to verify your identity immediately.")
explain_prediction(
    "Congratulations! You've been selected to receive a $1000 Walmart Gift Card. Click here to claim your reward!")
explain_prediction(
    "Invoice INV-99283 is overdue. Please download the attached PDF to avoid late fees and legal action.")
explain_prediction(
    "Dear Customer, we noticed suspicious activity on your credit card. Confirm your details now to prevent card blocking: http://bit.ly/bank-secure-auth")
explain_prediction(
    "I am a lawyer representing a deceased relative who left you $10.5M. Please reply with your bank details to initiate the transfer.")
explain_prediction(
    "Hi Laura how you doing. You paid for me last time so let me tak you for a dinner today. cant wait!")
explain_prediction(
    "Hi Claire, Thank you for your recent purchase. We are confirming that your order has been received and is being "
    "processed. You will receive a tracking number once the shipment is sent out. Please let us know if you have any "
    "questions. Best, Amy Johnson Customer Support Agent Online Store X")
explain_prediction(
    "I hope this email finds you well. I am checking on the status of the invoice I sent last week. Have you "
    "processed the payment? Best regards, Tom McNish Content Manager ABC Company")