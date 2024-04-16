import matplotlib.pyplot as plt

def save_plot(log_history, filename):
    epochs, f1_scores, precisions, recalls = [], [], [], []
    epoch = 0

    for log in log_history:
        if "eval_f1" in log and "eval_precision" in log and "eval_recall" in log:
            epoch += 1
            epochs.append(epoch)
            f1_scores.append(100 * log["eval_f1"])
            precisions.append(100 * log["eval_precision"])
            recalls.append(100 * log["eval_recall"])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, f1_scores, label="F1 Score")
    plt.plot(epochs, precisions, label="Precision")
    plt.plot(epochs, recalls, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value (%)")
    plt.title("Training Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
