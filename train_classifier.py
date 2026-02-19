"""
train_classifier.py — Train a ResNet50 freshness classifier (Fresh vs Rotten)
Run: python3 models/train_classifier.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time

# ============================================
# Configuration
# ============================================
DATA_DIR = os.path.expanduser("~/freshscan/data/processed")
SAVE_DIR = os.path.expanduser("~/freshscan/models")
NUM_CLASSES = 2  # fresh, rotten
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("  FRESHSCAN AI — CLASSIFIER TRAINING")
print("=" * 60)
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print("=" * 60)

# ============================================
# Data Transforms
# ============================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# Load Datasets
# ============================================
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=train_transforms
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"), transform=val_transforms
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

print(f"Classes: {train_dataset.classes}")
print(f"Class to index: {train_dataset.class_to_idx}")
print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")

# ============================================
# Model: Fine-tuned ResNet50
# ============================================
print("\nBuilding model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze early layers to speed up training
for param in list(model.parameters())[:-20]:
    param.requires_grad = False

# Replace the final classification layer
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)
print(f"Model loaded on {DEVICE}")

# ============================================
# Training Setup
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# ============================================
# Training Loop
# ============================================
print(f"\nStarting training for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # --- Training Phase ---
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

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    # --- Validation Phase ---
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

    # Calculate metrics
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    avg_train_loss = train_loss / train_total
    avg_val_loss = val_loss / val_total
    epoch_time = time.time() - epoch_start

    scheduler.step()

    # Save history
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
    print(f"  Train — Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} ({train_correct}/{train_total})")
    print(f"  Val   — Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} ({val_correct}/{val_total})")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(SAVE_DIR, "best_classifier.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
        }, save_path)
        print(f"  >> NEW BEST MODEL saved! (val_acc: {val_acc:.4f})")

    print()

# ============================================
# Final Summary
# ============================================
print("=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Best validation accuracy: {best_val_acc:.4f}")
print(f"  Model saved to: {os.path.join(SAVE_DIR, 'best_classifier.pth')}")
print("=" * 60)

# ============================================
# Quick Test on Test Set
# ============================================
print("\nRunning final test...")
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"), transform=val_transforms
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Load best model
checkpoint = torch.load(os.path.join(SAVE_DIR, "best_classifier.pth"), map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        test_correct += (outputs.argmax(1) == labels).sum().item()
        test_total += images.size(0)

test_acc = test_correct / test_total
print(f"  Test accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
print(f"\nDone! Your model is ready at: {SAVE_DIR}/best_classifier.pth")
