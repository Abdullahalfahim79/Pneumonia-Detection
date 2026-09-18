import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# CONFUSION MATRIX DATA
# ============================================================

# Class order:
# 0 = NORMAL
# 1 = PNEUMONIA

cm = np.array([
    [288, 29],
    [25, 830]
])


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# CREATE CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=True
)

plt.title("Confusion Matrix - ViT Pneumonia Detection")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()


# ============================================================
# SAVE IMAGE
# ============================================================

output_path = "results/confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix created successfully!")
print("Saved at:", output_path)