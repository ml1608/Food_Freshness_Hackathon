"""
train_classifier_v2.py — Train a ResNet50 that classifies BOTH fruit type AND freshness
This replaces the old 2-class (fresh/rotten) model with a 26-class model
(13 fruits × 2 freshness states = 26 classes like fresh_apple, rotten_banana, etc.)

Run: python3 models/train_classifier_v2.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

random.seed(42)

# ============================================
# Configuration
# ============================================
RAW_DIR = os.path.expanduser("~/freshscan/data/raw/Dataset")
DATA_DIR = os.path.expanduser("~/freshscan/data/processed_v2")
SAVE_DIR = os.path.expanduser("~/freshscan/models")
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
IMG_SIZE = 224
MAX_PER_FOLDER = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================
# Mapping from folder names to clean class names
# ============================================
FOLDER_MAP = {
    "FreshApple": "fresh_apple",
    "FreshBanana": "fresh_banana",
    "FreshOrange": "fresh_orange",
    "FreshTomato": "fresh_tomato",
    "FreshStrawberry": "fresh_strawberry",
    "FreshMango": "fresh_mango",
    "FreshPotato": "fresh_potato",
    "FreshCarrot": "fresh_carrot",
    "FreshCucumber": "fresh_cucumber",
    "FreshBellpepper": "fresh_bellpepper",
    "FreshCapciscum": "fresh_capsicum",
    "FreshOkara": "fresh_okra",
    "FreshBittergroud": "fresh_bittergourd",
    "RottenApple": "rotten_apple",
    "RottenBanana": "rotten_banana",
    "RottenOrange": "rotten_orange",
    "RottenTomato": "rotten_tomato",
    "RottenStrawberry": "rotten_strawberry",
    "RottenMango": "rotten_mango",
    "RottenPotato": "rotten_potato",
    "RottenCarrot": "rotten_carrot",
    "RottenCucumber": "rotten_cucumber",
    "RottenBellpepper": "rotten_bellpepper",
    "RottenCapsicum": "rotten_capsicum",
    "RottenOkra": "rotten_okra",
    "RottenBittergroud": "rotten_bittergourd",
}

# ============================================
# Step 1: Reorganize data into 26 classes
# ============================================
def organize_multiclass():
    """Create train/val/test with 26 class folders instead of 2."""
    print("=" * 60)
    print("  REORGANIZING DATA — 26 CLASSES")
    print("=" * 60)

    # Collect images per class
    class_images = {}

    for category in ["Fresh", "Rotten"]:
        cat_path = os.path.join(RAW_DIR, category)
        if not os.path.exists(cat_path):
            continue
        for folder in os.listdir(cat_path):
            folder_path = os.path.join(cat_path, folder)
            if not os.path.isdir(folder_path):
                continue
            class_name = FOLDER_MAP.get(folder)
            if not class_name:
                print(f"  WARNING: Skipping unknown folder '{folder}'")
                continue

            images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(images) > MAX_PER_FOLDER:
                images = random.sample(images, MAX_PER_FOLDER)

            class_images[class_name] = images
            print(f"  {class_name}: {len(images)} images")

    # Create directories and split
    for split in ["train", "val", "test"]:
        for cls in class_images:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    total = 0
    for cls, images in sorted(class_images.items()):
        train_imgs, temp = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest_dir = os.path.join(DATA_DIR, split_name, cls)
            for i, img_path in enumerate(split_imgs):
                ext = Path(img_path).suffix
                shutil.copy2(img_path, os.path.join(dest_dir, f"{cls}_{i:05d}{ext}"))
            total += len(split_imgs)

    print(f"\n  Total images organized: {total}")
    print(f"  Saved to: {DATA_DIR}")
    return True


# ============================================
# Step 2: Training
# ============================================
def train():
    print("\n" + "=" * 60)
    print("  FRESHSCAN AI v2 — MULTI-CLASS TRAINING")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_transforms)

    NUM_CLASSES = len(train_dataset.classes)
    print(f"\n  Number of classes: {NUM_CLASSES}")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    print(f"\n  Starting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"    Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        scheduler.step()

        print(f"\n  Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
        print(f"    Train — Loss: {train_loss/train_total:.4f} | Acc: {train_acc:.4f}")
        print(f"    Val   — Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
                'class_to_idx': train_dataset.class_to_idx,
                'num_classes': NUM_CLASSES,
            }, os.path.join(SAVE_DIR, "best_classifier_v2.pth"))
            print(f"    >> NEW BEST MODEL saved! (val_acc: {val_acc:.4f})")
        print()

    # Test
    print("=" * 60)
    print("  TRAINING COMPLETE — Running final test...")
    print("=" * 60)

    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_classifier_v2.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += images.size(0)

    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Test accuracy: {test_correct/test_total:.4f}")
    print(f"  Model saved: {SAVE_DIR}/best_classifier_v2.pth")
    print("=" * 60)


if __name__ == "__main__":
    # Check if data already organized
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        organize_multiclass()
    else:
        print("  Data already organized, skipping reorganization.")
        print(f"  (Delete {DATA_DIR} if you want to re-run)")

    train()
