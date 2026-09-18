import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Image transformations for ViT
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


class PneumoniaDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        classes = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

        for label_name, label in classes.items():

            folder = os.path.join(root_dir, label_name)

            if os.path.exists(folder):

                for img_name in os.listdir(folder):

                    img_path = os.path.join(folder, img_name)

                    self.images.append(img_path)
                    self.labels.append(label)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        img_path = self.images[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


# Test dataset loading
if __name__ == "__main__":

    print("Dataset configuration")
    print("---------------------")
    print("Class 0 : NORMAL")
    print("Class 1 : PNEUMONIA")
    print("Image size : 224 x 224")


    dataset_path = "dataset/train"

    dataset = PneumoniaDataset(
        dataset_path,
        transform=train_transform
    )

    print("\nTotal Images:", len(dataset))


    if len(dataset) > 0:
        image, label = dataset[0]

        print("Image Shape:", image.shape)
        print("Label:", label.item())