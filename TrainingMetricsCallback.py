from transformers import TrainerCallback
import matplotlib.pyplot as plt


class TrainingMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_metrics = []
        self.step_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            if state.global_step % 1000 == 0:
                self.step_losses.append({
                    "step": state.global_step,
                    "loss": logs["loss"]
                })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.epoch_metrics.append({
                "epoch": state.epoch,
                "val_loss": metrics.get("eval_loss", None),
                "accuracy": metrics.get("eval_accuracy", None),
                "f1": metrics.get("eval_f1", None),
                "precision": metrics.get("eval_precision", None),
                "recall": metrics.get("eval_recall", None),
            })


def plot_all_metrics(metrics_callback):
    step_data = metrics_callback.step_losses
    epoch_data = metrics_callback.epoch_metrics

    steps = [m["step"] for m in step_data]
    train_loss = [m["loss"] for m in step_data]

    epochs = [m["epoch"] for m in epoch_data]
    val_loss = [m["val_loss"] for m in epoch_data]
    accuracy = [m["accuracy"] for m in epoch_data]
    f1 = [m["f1"] for m in epoch_data]
    precision = [m["precision"] for m in epoch_data]
    recall = [m["recall"] for m in epoch_data]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Metryki: Loss (co 100 kroków) vs Metryki (co epokę)", fontsize=14)

    def _plot(ax, x, y, label, color, x_label="Epoka"):
        ax.plot(x, y, marker="o", color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel(x_label)
        ax.grid(True, linestyle="--", alpha=0.5)

    _plot(axes[0, 0], steps, train_loss, "Training Loss", "#E55A2B", x_label="Krok (Step)")
    _plot(axes[0, 1], epochs, val_loss, "Validation Loss", "#2B7BE5")
    _plot(axes[0, 2], epochs, accuracy, "Accuracy", "#27A06A")
    _plot(axes[1, 0], epochs, f1, "F1", "#9B59B6")
    _plot(axes[1, 1], epochs, precision, "Precision", "#E5A82B")
    _plot(axes[1, 2], epochs, recall, "Recall", "#E52B6A")

    plt.tight_layout()
    plt.savefig("detailed_metrics.png", dpi=150)
    plt.show()
