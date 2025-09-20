import os
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))  # Sort for consistent label mapping
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found or corrupted: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, label

# Transforms with augmentation (for train only)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),  # Added for more variety
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# For val/test: No augmentation, just resize and normalize
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Function to get loaders
def get_loaders(batch_size=32, num_workers=4):
    train_ds = PlantDataset('data/plantvillage-dataset/train', transform=train_transform)
    val_ds = PlantDataset('data/plantvillage-dataset/val', transform=val_transform)
    test_ds = PlantDataset('data/plantvillage-dataset/test', transform=val_transform)  # Added test for completeness
    
    if len(train_ds.classes) != 38:
        raise ValueError(f"Expected 38 classes, found {len(train_ds.classes)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_ds.classes  # Return classes for model/later use