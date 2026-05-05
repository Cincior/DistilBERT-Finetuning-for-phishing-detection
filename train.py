import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix
from training_metrics_callback import TrainingMetricsCallback, plot_all_metrics
from model_utils import compute_metrics
from config import SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY


def build_trainer(model, train_ds, val_ds):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
    )

    metrics_callback = TrainingMetricsCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )
    return trainer, metrics_callback


def evaluate_on_test(trainer, test_ds):
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

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Phishing'],
        yticklabels=['Normal', 'Phishing']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def run_training(model, tokenizer, df_train, df_val, df_test):
    from model_utils import tokenize_dataset, save_model

    train_ds = tokenize_dataset(df_train, tokenizer)
    val_ds = tokenize_dataset(df_val, tokenizer)
    test_ds = tokenize_dataset(df_test, tokenizer)

    trainer, metrics_callback = build_trainer(model, train_ds, val_ds)
    trainer.train()

    evaluate_on_test(trainer, test_ds)
    plot_all_metrics(metrics_callback)
    save_model(model, tokenizer)
