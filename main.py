from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import torch.nn.functional as F
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model_name = "distilbert-base-uncased"
save_path = "./saved_modelT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }


if os.path.exists(save_path):
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("csv", data_files="phishingEmail.csv")
    dataset = dataset.filter(lambda x: x["Email Text"] is not None and x["Email Type"] is not None)


    def preprocess(examples):
        tokenized = tokenizer(
            examples["Email Text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["labels"] = [int(examples["Email Type"][i]) for i in sample_mapping]
        return tokenized


    dataset = dataset.map(preprocess, remove_columns=["Email Text", "Email Type"], batched=True)

    train_test_split = dataset["train"].train_test_split(test_size=0.2, shuffle=True)
    test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, shuffle=True)

    train_ds = train_test_split["train"]
    val_ds = test_valid_split["train"]
    test_ds = test_valid_split["test"]

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
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
    print(confusion_matrix(y_true, y_pred))

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
            stride=128,
            return_overflowing_tokens=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "overflow_to_sample_mapping"}
        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

        avg_probs = np.mean(probs, axis=0)
        all_probs.append(avg_probs)
    return np.array(all_probs)


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
    print("\n" + "=" * 30)
    print("TEXT:", text[:100], "...")
    print(f"Normal: {round(probs[0] * 100, 2)}% | Phishing: {round(probs[1] * 100, 2)}%")
    exp = explainer.explain_instance(text, predict_proba, num_features=5, num_samples=500)
    fragments = get_suspicious_fragments(text, exp.as_list())
    print("\nSuspicious fragments:")
    for frag, score in fragments:
        print(f"[ {frag} ] -> {round(score, 3)}")


# =========================
# TEST
# =========================
explain_prediction("Dear Customer, We detected an unusual sign-in attempt on your account from a new device and "
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
    "Hi team, are we still meeting at 3 PM in the conference room? Let me know if we need to reschedule.")
