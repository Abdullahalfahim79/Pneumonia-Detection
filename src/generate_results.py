import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CREATE RESULTS FOLDER STRUCTURE
# ============================================================

results_dir = "results"

folders = [
    results_dir,
    os.path.join(results_dir, "graphs"),
    os.path.join(results_dir, "metrics"),
    os.path.join(results_dir, "reports"),
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Results folder structure created successfully!")


# ============================================================
# 2. ACTUAL TRAINING HISTORY
# ============================================================

epochs = list(range(1, 11))

train_loss = [
    0.2244,
    0.1455,
    0.1342,
    0.1201,
    0.1243,
    0.1151,
    0.1141,
    0.0993,
    0.1147,
    0.1020
]

train_accuracy = [
    90.90,
    94.34,
    95.10,
    95.66,
    95.54,
    95.68,
    95.71,
    96.34,
    96.00,
    96.24
]

val_accuracy = [
    94.70,
    94.36,
    95.56,
    96.07,
    93.33,
    94.70,
    95.73,
    94.19,
    95.90,
    95.21
]


# ============================================================
# 3. VALIDATION LOSS
# ============================================================
# Validation loss was not saved by the original train.py.
# Therefore, we do NOT invent validation-loss values.
#
# We will create the accuracy graph and training-loss graph
# using the actual recorded values.


# ============================================================
# 4. SAVE TRAINING HISTORY AS CSV
# ============================================================

history_df = pd.DataFrame({
    "Epoch": epochs,
    "Training Loss": train_loss,
    "Training Accuracy (%)": train_accuracy,
    "Validation Accuracy (%)": val_accuracy
})

history_path = os.path.join(
    results_dir,
    "metrics",
    "training_history.csv"
)

history_df.to_csv(history_path, index=False)

print("Training history saved:", history_path)


# ============================================================
# 5. FINAL TEST METRICS
# ============================================================

final_metrics = {
    "Accuracy": 95.39,
    "Precision": 96.62,
    "Recall": 97.08,
    "F1-score": 96.85
}


# Save metrics as JSON

metrics_json_path = os.path.join(
    results_dir,
    "metrics",
    "final_metrics.json"
)

with open(metrics_json_path, "w") as file:
    json.dump(final_metrics, file, indent=4)

print("Final metrics saved:", metrics_json_path)


# Save metrics as CSV

metrics_df = pd.DataFrame({
    "Metric": list(final_metrics.keys()),
    "Score (%)": list(final_metrics.values())
})

metrics_csv_path = os.path.join(
    results_dir,
    "metrics",
    "final_metrics.csv"
)

metrics_df.to_csv(metrics_csv_path, index=False)

print("Final metrics CSV saved:", metrics_csv_path)


# ============================================================
# 6. CONFUSION MATRIX DATA
# ============================================================

confusion_matrix = np.array([
    [288, 29],
    [25, 830]
])

cm_df = pd.DataFrame(
    confusion_matrix,
    index=["NORMAL", "PNEUMONIA"],
    columns=["NORMAL", "PNEUMONIA"]
)

cm_path = os.path.join(
    results_dir,
    "metrics",
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_path)

print("Confusion matrix data saved:", cm_path)


# ============================================================
# 7. GRAPH 1 - TRAINING VS VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.ylim(85, 100)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_graph_path = os.path.join(
    results_dir,
    "graphs",
    "training_validation_accuracy.png"
)

plt.savefig(
    accuracy_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Accuracy graph saved:", accuracy_graph_path)


# ============================================================
# 8. GRAPH 2 - TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

loss_graph_path = os.path.join(
    results_dir,
    "graphs",
    "training_loss.png"
)

plt.savefig(
    loss_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training loss graph saved:", loss_graph_path)


# ============================================================
# 9. GRAPH 3 - FINAL MODEL METRICS
# ============================================================

plt.figure(figsize=(9, 6))

metric_names = list(final_metrics.keys())
metric_values = list(final_metrics.values())

bars = plt.bar(
    metric_names,
    metric_values
)

plt.title("Final ViT Pneumonia Detection Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

# Display value above each bar

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

metrics_graph_path = os.path.join(
    results_dir,
    "graphs",
    "final_metrics.png"
)

plt.savefig(
    metrics_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Final metrics graph saved:", metrics_graph_path)


# ============================================================
# 10. CREATE FINAL TEXT REPORT
# ============================================================

report_path = os.path.join(
    results_dir,
    "reports",
    "final_model_report.txt"
)

with open(report_path, "w") as file:

    file.write("PNEUMONIA DETECTION - FINAL MODEL REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write("Model: Vision Transformer (ViT-B/16)\n")
    file.write("Dataset: Chest X-ray Images\n")
    file.write("Classes: NORMAL, PNEUMONIA\n")
    file.write("Total Images: 5856\n")
    file.write("Training: 70% (4099 images)\n")
    file.write("Validation: 10% (585 images)\n")
    file.write("Testing: 20% (1172 images)\n")
    file.write("Image Size: 224 x 224\n")
    file.write("Batch Size: 8\n")
    file.write("Epochs: 10\n")
    file.write("Optimizer: AdamW\n")
    file.write("Loss Function: Cross Entropy Loss\n")
    file.write("Device: CPU\n\n")

    file.write("BEST VALIDATION RESULT\n")
    file.write("-" * 50 + "\n")
    file.write("Best Epoch: 4\n")
    file.write("Best Validation Accuracy: 96.07%\n\n")

    file.write("FINAL TEST RESULT\n")
    file.write("-" * 50 + "\n")

    for metric, value in final_metrics.items():
        file.write(f"{metric}: {value:.2f}%\n")

    file.write("\nCONFUSION MATRIX\n")
    file.write("-" * 50 + "\n")
    file.write("                 Predicted\n")
    file.write("              NORMAL  PNEUMONIA\n")
    file.write("Actual NORMAL     288       29\n")
    file.write("Actual PNEUMONIA   25      830\n")


print("Final report saved:", report_path)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("RESULTS GENERATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")

print("\nGraphs:")
print("1.", accuracy_graph_path)
print("2.", loss_graph_path)
print("3.", metrics_graph_path)

print("\nMetrics:")
print("4.", history_path)
print("5.", metrics_json_path)
print("6.", metrics_csv_path)
print("7.", cm_path)

print("\nReport:")
print("8.", report_path)