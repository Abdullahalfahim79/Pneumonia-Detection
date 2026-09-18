import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import PneumoniaDataset, val_test_transform
from model import PneumoniaViT


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PNEUMONIA DETECTION - TEST EVALUATION")
print("=" * 60)

print("Testing device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. TEST DATASET
# ============================================================

test_path = "dataset/test"

test_dataset = PneumoniaDataset(
    test_path,
    transform=val_test_transform
)

print("\nTest Dataset Loaded")
print("Test images:", len(test_dataset))


# ============================================================
# 3. TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Test batches:", len(test_loader))


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nLoading best trained ViT model...")

model = PneumoniaViT(num_classes=2)

model_path = "models/best_vit_pneumonia.pth"

if not os.path.exists(model_path):
    print("\nERROR: Model file not found!")
    print("Expected location:", model_path)
    exit()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Best model loaded successfully!")
print("Model:", model_path)


# ============================================================
# 5. TESTING
# ============================================================

print("\n" + "=" * 60)
print("TESTING STARTED")
print("=" * 60)

all_labels = []
all_predictions = []

with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Testing",
        unit="batch"
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Prediction
        _, predicted = torch.max(outputs, 1)

        # Store labels and predictions
        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# 8. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "Test Accuracy : {:.2f}%".format(
        accuracy * 100
    )
)

print(
    "Precision     : {:.2f}%".format(
        precision * 100
    )
)

print(
    "Recall        : {:.2f}%".format(
        recall * 100
    )
)

print(
    "F1-Score      : {:.2f}%".format(
        f1 * 100
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nClass order:")
print("0 = NORMAL")
print("1 = PNEUMONIA")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ],
        zero_division=0
    )
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results_file = "results/test_results.txt"

with open(results_file, "w") as f:

    f.write("PNEUMONIA DETECTION - TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        "Test Images: {}\n\n".format(
            len(test_dataset)
        )
    )

    f.write(
        "Test Accuracy: {:.2f}%\n".format(
            accuracy * 100
        )
    )

    f.write(
        "Precision: {:.2f}%\n".format(
            precision * 100
        )
    )

    f.write(
        "Recall: {:.2f}%\n".format(
            recall * 100
        )
    )

    f.write(
        "F1-Score: {:.2f}%\n\n".format(
            f1 * 100
        )
    )

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report:\n")

    f.write(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "NORMAL",
                "PNEUMONIA"
            ],
            zero_division=0
        )
    )


print("\nResults saved to:")
print(results_file)

print("\n" + "=" * 60)
print("TESTING FINISHED")
print("=" * 60)