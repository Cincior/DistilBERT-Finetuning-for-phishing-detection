from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn.functional as F
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model_name = "allenai/longformer-base-4096"
save_path = "./saved_model_longformer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================
# LOAD OR TRAIN MODEL
# =========================
if os.path.exists(save_path):
    print("Wczytywanie zapisanego modelu...")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device)

else:
    print("Trenowanie modelu...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset(
        "csv",
        data_files="phishing_email.csv",
        delimiter=",",
        column_names=["text", "label"]
    )

    dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)

    def preprocess(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=4096
        )
        tokens["labels"] = [int(label) for label in examples["label"]]
        return tokens

    dataset = dataset.map(preprocess, remove_columns=["text", "label"], batched=True)
    dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()
    print("\n=== METRYKI ===")
    for key, value in results.items():
        print(f"{key}: {round(value, 4)}")

    predictions = trainer.predict(dataset["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.to(device)


# =========================
# PROBABILITIES
# =========================
def predict_proba(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=4096
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()


# =========================
# LIME
# =========================
explainer = LimeTextExplainer(class_names=["normal", "phishing"])


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


def explain_prediction(text):
    probs = predict_proba([text])[0]

    print("\n==============================")
    print("TEXT:", text)
    print("Normal:", round(probs[0] * 100, 2), "%")
    print("Phishing:", round(probs[1] * 100, 2), "%")

    if probs[1] > probs[0]:
        print("Prediction: PHISHING")
    else:
        print("Prediction: NORMAL")

    exp = explainer.explain_instance(text, predict_proba, num_features=3, num_samples=1000)
    lime_words = exp.as_list()

    fragments = get_suspicious_fragments(text, lime_words)

    print("\nSuspicious fragments:")
    for frag, score in fragments:
        print(f"[ {frag} ] -> {round(score, 3)}")


# =========================
# TEST
# =========================
explain_prediction("Dear Customer, We detected an unusual sign-in attempt on your account from a new device and location...")
explain_prediction("URGENT: Your PayPal account has been suspended. Please log in immediately...")
explain_prediction("Congratulations! You've been selected to receive a $1000 Walmart Gift Card...")
explain_prediction("Invoice INV-99283 is overdue. Please download the attached PDF...")
explain_prediction("Dear Customer, we noticed suspicious activity on your credit card...")
explain_prediction("I am a lawyer representing a deceased relative who left you $10.5M...")
explain_prediction("Hi team, are we still meeting at 3 PM in the conference room?")


# =========================
# TEST
# =========================
explain_prediction("Dear Customer, We detected an unusual sign-in attempt on your account from a new device and " "location. For your security, your account has been temporarily limited. Please confirm your " "identity and restore full access by clicking the secure link below: Verify My Account Now → " "http://security-check-account.com/verify If you do not verify your account within 24 hours, " "your account will be permanently suspended. Thank you for your cooperation, Security Team") explain_prediction("URGENT: Your PayPal account has been suspended. Please log in at http://secure-paypal-login.com to verify your identity immediately.")
explain_prediction("URGENT: Your PayPal account has been suspended. Please log in immediately...")
explain_prediction("Congratulations! You've been selected to receive a $1000 Walmart Gift Card...")
explain_prediction("Invoice INV-99283 is overdue. Please download the attached PDF...")
explain_prediction("Dear Customer, we noticed suspicious activity on your credit card...")
explain_prediction("I am a lawyer representing a deceased relative who left you $10.5M...")
explain_prediction("Hi team, are we still meeting at 3 PM in the conference room?")