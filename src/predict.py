import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from model import PneumoniaViT


# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_vit_pneumonia.pth"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. Image preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. Load trained model
# ==========================================

print("\nLoading Pneumonia Detection Model...")

model = PneumoniaViT(num_classes=2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:
        model.load_state_dict(checkpoint)

else:
    model.load_state_dict(checkpoint)


model.to(DEVICE)
model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


# ==========================================
# 4. Prediction function
# ==========================================

def predict_image(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU
    image_tensor = image_tensor.to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted_class.item()

    confidence = confidence.item() * 100

    prediction = CLASS_NAMES[predicted_class]

    return prediction, confidence


# ==========================================
# 5. Main program
# ==========================================

if __name__ == "__main__":

    print("\n")
    print("=" * 55)
    print("        PNEUMONIA DETECTION SYSTEM")
    print("=" * 55)

    print("\nModel      : Vision Transformer (ViT-B/16)")
    print("Image Size : 224 x 224")
    print("Classes    : NORMAL / PNEUMONIA")

    image_path = input(
        "\nEnter the path of the chest X-ray image: "
    ).strip().strip('"')

    image_path = Path(image_path)

    if not image_path.exists():

        print("\nERROR: Image file not found.")

        print(
            "Please enter a valid image path."
        )

    else:

        try:

            prediction, confidence = predict_image(
                image_path
            )

            print("\n")
            print("=" * 55)
            print("                    RESULT")
            print("=" * 55)

            print(
                f"\nPrediction : {prediction}"
            )

            print(
                f"Confidence : {confidence:.2f}%"
            )

            print("\n" + "=" * 55)

        except Exception as e:

            print("\nERROR during prediction:")
            print(e)