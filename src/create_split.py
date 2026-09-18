from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

SOURCE_DIR = Path("archive/chest_xray/chest_xray")
OUTPUT_DIR = Path("dataset")

CLASSES = ["NORMAL", "PNEUMONIA"]

TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20

RANDOM_STATE = 42

# ============================================================
# CHECK SETTINGS
# ============================================================

assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == 1.0

print("Source dataset:")
print(SOURCE_DIR.resolve())

print("\nOutput dataset:")
print(OUTPUT_DIR.resolve())

# ============================================================
# COLLECT ALL IMAGES
# ============================================================

image_extensions = {".jpg", ".jpeg", ".png"}

all_images = []
all_labels = []

for class_name in CLASSES:
    class_path = SOURCE_DIR / "train" / class_name

    if not class_path.exists():
        print(f"ERROR: Folder not found: {class_path}")
        raise SystemExit

    for image_path in class_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            all_images.append(image_path)
            all_labels.append(class_name)

    class_path = SOURCE_DIR / "test" / class_name

    if class_path.exists():
        for image_path in class_path.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                all_images.append(image_path)
                all_labels.append(class_name)

    class_path = SOURCE_DIR / "val" / class_name

    if class_path.exists():
        for image_path in class_path.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                all_images.append(image_path)
                all_labels.append(class_name)

print("\nTotal images found:", len(all_images))

for class_name in CLASSES:
    count = all_labels.count(class_name)
    print(f"{class_name}: {count}")

# ============================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

train_images, temp_images, train_labels, temp_labels = train_test_split(
    all_images,
    all_labels,
    test_size=(VAL_SIZE + TEST_SIZE),
    stratify=all_labels,
    random_state=RANDOM_STATE
)

# ============================================================
# SECOND SPLIT
# TEMPORARY -> 10% VALIDATION + 20% TEST
#
# 10 / 30 = 1/3
# ============================================================

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
    stratify=temp_labels,
    random_state=RANDOM_STATE
)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# COPY FILES
# ============================================================

def copy_images(images, labels, split_name):
    for image_path, label in zip(images, labels):

        destination_folder = OUTPUT_DIR / split_name / label
        destination = destination_folder / image_path.name

        # Prevent filename collision
        counter = 1

        while destination.exists():
            destination = (
                destination_folder
                / f"{image_path.stem}_{counter}{image_path.suffix}"
            )
            counter += 1

        shutil.copy2(image_path, destination)


print("\nCopying training images...")
copy_images(train_images, train_labels, "train")

print("Copying validation images...")
copy_images(val_images, val_labels, "val")

print("Copying testing images...")
copy_images(test_images, test_labels, "test")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("DATASET SPLIT COMPLETED")
print("========================================")

print(f"Train:      {len(train_images)} images")
print(f"Validation: {len(val_images)} images")
print(f"Test:       {len(test_images)} images")
print(f"Total:      {len(train_images) + len(val_images) + len(test_images)} images")

print("\nDataset created at:")
print(OUTPUT_DIR.resolve())

print("\nStructure:")
print("dataset/")
print("├── train/")
print("│   ├── NORMAL/")
print("│   └── PNEUMONIA/")
print("├── val/")
print("│   ├── NORMAL/")
print("│   └── PNEUMONIA/")
print("└── test/")
print("    ├── NORMAL/")
print("    └── PNEUMONIA/")