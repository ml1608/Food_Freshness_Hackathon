"""
evaluate.py — Full evaluation with precision, recall, F1, confusion matrix
Run: python3 models/evaluate.py
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

DATA_DIR = os.path.expanduser("~/freshscan/data/processed")
MODEL_PATH = os.path.expanduser("~/freshscan/models/best_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as training validation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"), transform=val_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Test set: {len(test_dataset)} images")
print(f"Classes: {test_dataset.classes}")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Collect predictions
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================
# Results
# ============================================
print("\n" + "=" * 60)
print("  FULL EVALUATION RESULTS")
print("=" * 60)

# Classification Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=test_dataset.classes,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(f"                Predicted Fresh  Predicted Rotten")
print(f"  Actual Fresh:     {cm[0][0]:>6}          {cm[0][1]:>6}")
print(f"  Actual Rotten:    {cm[1][0]:>6}          {cm[1][1]:>6}")

# Key metrics for judges
total = len(all_labels)
correct = (all_preds == all_labels).sum()
accuracy = correct / total

# False positive rate (fresh classified as rotten — unnecessary waste)
fresh_as_rotten = cm[0][1]
# False negative rate (rotten classified as fresh — DANGEROUS)
rotten_as_fresh = cm[1][0]

print(f"\n  Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"\n  CRITICAL SAFETY METRICS:")
print(f"  Fresh misclassified as Rotten: {fresh_as_rotten} (unnecessary waste)")
print(f"  Rotten misclassified as Fresh: {rotten_as_fresh} (SAFETY RISK)")
print(f"\n  False Negative Rate (rotten missed): {rotten_as_fresh/cm[1].sum():.4f}")
print(f"  This means {rotten_as_fresh} rotten items could reach customers")
print("=" * 60)
