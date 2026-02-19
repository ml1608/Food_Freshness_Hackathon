"""
train_swinv2.py — Train SwinV2-Tiny for freshness classification
SwinV2's shifted window attention captures fine texture details
(bruising, discoloration, surface decay) much better than ResNet50.

Run: python3 models/train_swinv2.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, confusion_matrix
import timm
import os
import time
import numpy as np

# ============================================
# Configuration
# ============================================
DATA_DIR = os.path.expanduser("~/freshscan/data/processed")
SAVE_DIR = os.path.expanduser("~/freshscan/models")
BATCH_SIZE = 24          # Slightly smaller for SwinV2 memory
EPOCHS = 25
LEARNING_RATE = 3e-5     # Lower LR for transformer fine-tuning
WEIGHT_DECAY = 0.05      # Important for transformers
IMG_SIZE = 256           # SwinV2 benefits from slightly larger images
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("  FRESHSCAN AI — SWINV2-TINY TRAINING")
print("  Deep Texture Analysis for Freshness Detection")
print("=" * 60)
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Model: SwinV2-Tiny (28M params)")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print("=" * 60)

# ============================================
# Data Transforms — Enhanced for texture analysis
# ============================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize larger first
    transforms.RandomCrop(IMG_SIZE),                     # Then crop for variety
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.25, contrast=0.25,
        saturation=0.20, hue=0.08
    ),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomGrayscale(p=0.05),  # Occasionally remove color to focus on texture
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Random occlusion
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# Load Data
# ============================================
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"  Classes: {train_dataset.classes}")
print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ============================================
# Build SwinV2-Tiny Model
# ============================================
print("\nLoading SwinV2-Tiny pretrained model...")
model = timm.create_model(
    'swinv2_tiny_window8_256',
    pretrained=True,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
)

# Freeze first 2 stages (low-level features) to prevent overfitting
# Stages: patch_embed -> layers[0] -> layers[1] -> layers[2] -> layers[3] -> head
freeze_stages = 2
frozen_count = 0
for name, param in model.named_parameters():
    if any(f"layers.{i}." in name for i in range(freeze_stages)):
        param.requires_grad = False
        frozen_count += 1
    elif "patch_embed" in name:
        param.requires_grad = False
        frozen_count += 1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
print(f"  Frozen: {frozen_count} parameter groups (stages 0-{freeze_stages-1})")

model = model.to(DEVICE)

# ============================================
# Training Setup
# ============================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps transformers
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

best_val_acc = 0.0
patience = 7
patience_counter = 0

# ============================================
# Training Loop
# ============================================
print(f"\nStarting training...\n")

for epoch in range(EPOCHS):
    t0 = time.time()

    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"    Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_probs_all = []
    val_labels_all = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            val_correct += (probs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

            val_probs_all.extend(probs.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    elapsed = time.time() - t0
    scheduler.step()

    print(f"\n  Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s)")
    print(f"    Train — Loss: {train_loss/train_total:.4f} | Acc: {train_acc:.4f}")
    print(f"    Val   — Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.4f}")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        # Calculate per-class metrics for checkpoint
        val_probs_np = np.array(val_probs_all)
        val_labels_np = np.array(val_labels_all)
        val_preds = val_probs_np.argmax(axis=1)

        # Average confidence for fresh and rotten predictions
        fresh_mask = val_preds == train_dataset.class_to_idx.get("fresh", 0)
        rotten_mask = val_preds == train_dataset.class_to_idx.get("rotten", 1)

        avg_fresh_conf = val_probs_np[fresh_mask].max(axis=1).mean() if fresh_mask.any() else 0
        avg_rotten_conf = val_probs_np[rotten_mask].max(axis=1).mean() if rotten_mask.any() else 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx,
            'model_name': 'swinv2_tiny_window8_256',
            'img_size': IMG_SIZE,
            'avg_fresh_confidence': float(avg_fresh_conf),
            'avg_rotten_confidence': float(avg_rotten_conf),
        }, os.path.join(SAVE_DIR, "best_classifier.pth"))
        print(f"    >> BEST MODEL saved (val_acc: {val_acc:.4f})")
        print(f"       Avg fresh confidence: {avg_fresh_conf:.4f}")
        print(f"       Avg rotten confidence: {avg_rotten_conf:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print()

# ============================================
# Final Test with Full Metrics
# ============================================
print("=" * 60)
print("  FINAL EVALUATION ON TEST SET")
print("=" * 60)

checkpoint = torch.load(os.path.join(SAVE_DIR, "best_classifier.pth"), map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=test_dataset.classes, digits=4))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(f"                 Predicted Fresh  Predicted Rotten")
print(f"  Actual Fresh:      {cm[0][0]:>6}          {cm[0][1]:>6}")
print(f"  Actual Rotten:     {cm[1][0]:>6}          {cm[1][1]:>6}")

rotten_as_fresh = cm[1][0]
total_rotten = cm[1].sum()
print(f"\n  SAFETY: Rotten missed as Fresh: {rotten_as_fresh}/{total_rotten} "
      f"({rotten_as_fresh/total_rotten*100:.2f}%)")

test_acc = (all_preds == all_labels).sum() / len(all_labels)
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"\n  Model saved: {SAVE_DIR}/best_classifier.pth")
print("=" * 60)
