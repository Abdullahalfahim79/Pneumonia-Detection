import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_vit_pneumonia.pth"
SRC_DIR = BASE_DIR / "src"

# Add src folder to Python path
sys.path.insert(0, str(SRC_DIR))

from model import PneumoniaViT

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = PneumoniaViT(num_classes=2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        else:
            model.load_state_dict(checkpoint)

    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image, model):

    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100

    prediction = CLASS_NAMES[predicted_class]

    return prediction, confidence

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🫁 Pneumonia Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Vision Transformer (ViT-B/16) Based Chest X-ray Classification
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("📊 Model Information"):

    st.write("**Model:** Vision Transformer (ViT-B/16)")
    st.write("**Dataset:** Chest X-ray Images")
    st.write("**Classes:** NORMAL, PNEUMONIA")
    st.write("**Image Size:** 224 × 224")
    st.write("**Training Images:** 4099")
    st.write("**Validation Images:** 585")
    st.write("**Testing Images:** 1172")

    st.write("")

    st.write("**Test Accuracy:** 95.39%")
    st.write("**Precision:** 96.62%")
    st.write("**Recall:** 97.08%")
    st.write("**F1-score:** 96.85%")

    st.write("")

    st.write(f"**Device:** {DEVICE}")

# ============================================================
# UPLOAD X-RAY
# ============================================================

st.subheader("📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# DISPLAY AND PREDICT
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("🩻 Uploaded X-ray")

    st.image(
        image,
        caption="Chest X-ray",
        use_container_width=True
    )

    st.write("")

    predict_button = st.button(
        "🔍 Predict",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        with st.spinner("Analyzing chest X-ray..."):

            try:

                model = load_model()

                prediction, confidence = predict_image(
                    image,
                    model
                )

                st.divider()

                st.subheader("📋 Prediction Result")

                if prediction == "PNEUMONIA":

                    st.error(f"Prediction: {prediction}")

                else:

                    st.success(f"Prediction: {prediction}")

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.write("")

                st.info(
                    "This application is developed for academic and "
                    "demonstration purposes. It should not be used as "
                    "a substitute for professional medical diagnosis."
                )

            except Exception as e:

                st.error(
                    "An error occurred while processing the image."
                )

                st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pneumonia Detection System | "
    "Vision Transformer (ViT-B/16) | "
    "Academic Project"
)