"""
train_fruit_verifier.py — Train a lightweight fruit-type verifier
This model ONLY identifies fruit type (13 classes), NOT freshness.
Used to verify that the shopkeeper selected the correct dropdown option.

Uses EfficientNet-B0 for speed (runs alongside SwinV2 without slowdown).

Run: python3 models/train_fruit_verifier.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import os
import time
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

# ============================================
# Configuration
# ============================================
RAW_DIR = os.path.expanduser("~/freshscan/data/raw/Dataset")
DATA_DIR = os.path.expanduser("~/freshscan/data/fruit_type")
SAVE_DIR = os.path.expanduser("~/freshscan/models")
BATCH_SIZE = 32
EPOCHS = 15
LR = 2e-4
IMG_SIZE = 224
MAX_PER_CLASS = 1500  # Keep it fast
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# Folder name → clean fruit type (merging fresh + rotten into same type)
FOLDER_MAP = {
    "FreshApple": "apple", "RottenApple": "apple",
    "FreshBanana": "banana", "RottenBanana": "banana",
    "FreshOrange": "orange", "RottenOrange": "orange",
    "FreshTomato": "tomato", "RottenTomato": "tomato",
    "FreshStrawberry": "strawberry", "RottenStrawberry": "strawberry",
    "FreshMango": "mango", "RottenMango": "mango",
    "FreshPotato": "potato", "RottenPotato": "potato",
    "FreshCarrot": "carrot", "RottenCarrot": "carrot",
    "FreshCucumber": "cucumber", "RottenCucumber": "cucumber",
    "FreshBellpepper": "bellpepper", "RottenBellpepper": "bellpepper",
    "FreshCapciscum": "capsicum", "RottenCapsicum": "capsicum",
    "FreshOkara": "okra", "RottenOkra": "okra",
    "FreshBittergroud": "bittergourd", "RottenBittergroud": "bittergourd",
}

print("=" * 60)
print("  FRUIT TYPE VERIFIER — TRAINING")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ============================================
# Organize data by fruit type (ignoring freshness)
# ============================================
def organize():
    print("\nOrganizing data by fruit type...")
    class_images = {}

    for category in ["Fresh", "Rotten"]:
        cat_path = os.path.join(RAW_DIR, category)
        if not os.path.exists(cat_path):
            continue
        for folder in os.listdir(cat_path):
            folder_path = os.path.join(cat_path, folder)
            if not os.path.isdir(folder_path):
                continue
            fruit = FOLDER_MAP.get(folder)
            if not fruit:
                continue

            if fruit not in class_images:
                class_images[fruit] = []

            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_images[fruit].append(os.path.join(folder_path, f))

    # Cap and split
    for split in ["train", "val"]:
        for fruit in class_images:
            os.makedirs(os.path.join(DATA_DIR, split, fruit), exist_ok=True)

    for fruit, images in sorted(class_images.items()):
        if len(images) > MAX_PER_CLASS * 2:
            images = random.sample(images, MAX_PER_CLASS * 2)

        train_imgs, val_imgs = train_test_split(images, test_size=0.15, random_state=42)

        for i, img in enumerate(train_imgs):
            ext = Path(img).suffix
            shutil.copy2(img, os.path.join(DATA_DIR, "train", fruit, f"{fruit}_{i:05d}{ext}"))
        for i, img in enumerate(val_imgs):
            ext = Path(img).suffix
            shutil.copy2(img, os.path.join(DATA_DIR, "val", fruit, f"{fruit}_{i:05d}{ext}"))

        print(f"  {fruit}: train={len(train_imgs)}, val={len(val_imgs)}")


# ============================================
# Train
# ============================================
def train():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    from torchvision import datasets
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_tf)
    num_classes = len(train_ds.classes)

    print(f"\n  Classes ({num_classes}): {train_ds.classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # EfficientNet-B0: fast, lightweight, good enough for type verification
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_acc = 0
    for epoch in range(EPOCHS):
        t0 = time.time()

        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images)
                vc += (out.argmax(1) == labels).sum().item()
                vt += images.size(0)

        ta, va = correct/total, vc/vt
        print(f"  Epoch {epoch+1}/{EPOCHS} ({time.time()-t0:.1f}s) | "
              f"Train: {ta:.4f} | Val: {va:.4f}", end="")

        if va > best_acc:
            best_acc = va
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_ds.classes,
                'class_to_idx': train_ds.class_to_idx,
                'val_acc': va,
            }, os.path.join(SAVE_DIR, "fruit_verifier.pth"))
            print(f" >> SAVED", end="")
        print()

    print(f"\n  Best accuracy: {best_acc:.4f}")
    print(f"  Saved: {SAVE_DIR}/fruit_verifier.pth")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        organize()
    train()
