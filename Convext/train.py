import os
import json
import timm
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_DIR = 'data'
IMG_DIR = os.path.join(DATA_DIR, 'train_images')
CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_num_to_disease_map.json')
BATCH_SIZE = 16
NUM_CLASSES = 5
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row.image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label

        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# Transforms
train_transform = A.Compose([
    A.Resize(512, 512),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * y.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

def main():
    # Load data
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df.label, random_state=42)

    train_ds = CassavaDataset(train_df, IMG_DIR, transform=train_transform)
    val_ds = CassavaDataset(val_df, IMG_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model: ED-Swin Transformer
    model = timm.create_model("convnextv2_tiny", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    # Optimizer, Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, eps=1e-6)
    warm_restart_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=LR / 7)

    criterion = nn.CrossEntropyLoss()

    best_score = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            # Warm restart step per batch
            warm_restart_scheduler.step(epoch + step / len(train_loader))
            train_loss += loss.item() * y.size(0)

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        plateau_scheduler.step(val_loss)
        if val_acc > best_score:
            best_score = val_acc
            torch.save(model.state_dict(), "convnextv2_small_cassava.pth")
            print(f"Saved best model with score {val_acc:.4f}")

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_ds):.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

# Windows-safe multiprocessing protection
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
