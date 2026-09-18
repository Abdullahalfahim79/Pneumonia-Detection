from torchvision import transforms


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Data augmentation - training only
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if __name__ == "__main__":
    print("Preprocessing configuration")
    print("---------------------------")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Training: augmentation + normalization")
    print("Validation: resize + normalization")
    print("Test: resize + normalization")