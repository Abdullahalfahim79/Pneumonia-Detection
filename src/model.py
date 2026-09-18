import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class PneumoniaViT(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaViT, self).__init__()

        # Load pretrained ViT-B/16
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)

        # Replace classifier head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("Model configuration")
    print("-------------------")
    print("Architecture : ViT-B/16")
    print("Pretrained   : Yes (ImageNet)")
    print("Image Size   : 224 x 224")
    print("Classes      : 2")
    print("Class 0      : NORMAL")
    print("Class 1      : PNEUMONIA")

    model = PneumoniaViT(num_classes=2)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("\nInput Shape :", x.shape)
    print("Output Shape:", y.shape)