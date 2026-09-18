import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import PneumoniaDataset, train_transform, val_test_transform
from model import PneumoniaViT


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PNEUMONIA DETECTION - ViT TRAINING")
print("=" * 60)

print("Training device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. DATASET PATHS
# ============================================================

train_path = "dataset/train"
val_path = "dataset/val"


# ============================================================
# 3. LOAD DATASETS
# ============================================================

train_dataset = PneumoniaDataset(
    train_path,
    transform=train_transform
)

val_dataset = PneumoniaDataset(
    val_path,
    transform=val_test_transform
)

print("\nDataset Loaded")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# 4. DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\nLoading Vision Transformer model...")

model = PneumoniaViT(num_classes=2)

model = model.to(device)

print("Model loaded successfully!")


# ============================================================
# 6. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 7. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01
)


# ============================================================
# 8. TRAINING SETTINGS
# ============================================================

epochs = 10

best_accuracy = 0.0

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

best_model_path = "models/best_vit_pneumonia.pth"


# ============================================================
# 9. TRAINING START
# ============================================================

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)


for epoch in range(epochs):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print("\nEpoch {}/{}".format(epoch + 1, epochs))

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        unit="batch"
    )

    for batch_number, (images, labels) in enumerate(progress_bar):

        # Move data to CPU/GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model weights
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        current_accuracy = 100.0 * correct / total

        # Show live progress
        progress_bar.set_postfix(
            loss="{:.4f}".format(loss.item()),
            accuracy="{:.2f}%".format(current_accuracy)
        )


    # --------------------------------------------------------
    # TRAINING RESULTS
    # --------------------------------------------------------

    train_accuracy = 100.0 * correct / total

    average_loss = running_loss / len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nRunning validation...")

    model.eval()

    val_correct = 0
    val_total = 0

    val_progress = tqdm(
        val_loader,
        desc="Validation",
        unit="batch"
    )

    with torch.no_grad():

        for images, labels in val_progress:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()


    # --------------------------------------------------------
    # VALIDATION ACCURACY
    # --------------------------------------------------------

    val_accuracy = 100.0 * val_correct / val_total


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_time = time.time() - epoch_start


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("Epoch {} completed".format(epoch + 1))
    print("-" * 60)

    print("Loss: {:.4f}".format(average_loss))

    print(
        "Training Accuracy: {:.2f}%".format(train_accuracy)
    )

    print(
        "Validation Accuracy: {:.2f}%".format(val_accuracy)
    )

    print(
        "Epoch Time: {:.2f} minutes".format(epoch_time / 60)
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("\nBest model saved!")
        print("Location:", best_model_path)


# ============================================================
# 10. TRAINING FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    "Best Validation Accuracy: {:.2f}%".format(best_accuracy)
)

print(
    "Best model location:",
    best_model_path
)