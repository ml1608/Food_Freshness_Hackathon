"""
organize_data.py — Organizes the Food Freshness Dataset into train/val/test splits
Run: python3 utils/organize_data.py
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

random.seed(42)

RAW_DIR = os.path.expanduser("~/freshscan/data/raw/Dataset")
OUT_DIR = os.path.expanduser("~/freshscan/data/processed")

# Mapping from dataset folder names to clean fruit type names
FRUIT_MAP = {
    "FreshApple": ("apple", "fresh"),
    "FreshBanana": ("banana", "fresh"),
    "FreshOrange": ("orange", "fresh"),
    "FreshTomato": ("tomato", "fresh"),
    "FreshStrawberry": ("strawberry", "fresh"),
    "FreshMango": ("mango", "fresh"),
    "FreshPotato": ("potato", "fresh"),
    "FreshCarrot": ("carrot", "fresh"),
    "FreshCucumber": ("cucumber", "fresh"),
    "FreshBellpepper": ("bellpepper", "fresh"),
    "FreshCapciscum": ("capsicum", "fresh"),
    "FreshOkara": ("okra", "fresh"),
    "FreshBittergroud": ("bittergourd", "fresh"),
    "RottenApple": ("apple", "rotten"),
    "RottenBanana": ("banana", "rotten"),
    "RottenOrange": ("orange", "rotten"),
    "RottenTomato": ("tomato", "rotten"),
    "RottenStrawberry": ("strawberry", "rotten"),
    "RottenMango": ("mango", "rotten"),
    "RottenPotato": ("potato", "rotten"),
    "RottenCarrot": ("carrot", "rotten"),
    "RottenCucumber": ("cucumber", "rotten"),
    "RottenBellpepper": ("bellpepper", "rotten"),
    "RottenCapsicum": ("capsicum", "rotten"),
    "RottenOkra": ("okra", "rotten"),
    "RottenBittergroud": ("bittergourd", "rotten"),
}

# We cap each class to avoid massive imbalance
# (FreshTomato has 13,679 but FreshBittergroud only 327)
MAX_IMAGES_PER_FOLDER = 2000

def main():
    # Collect all images grouped by freshness class
    fresh_images = []
    rotten_images = []
    
    fruit_stats = {}

    for category in ["Fresh", "Rotten"]:
        category_path = os.path.join(RAW_DIR, category)
        if not os.path.exists(category_path):
            print(f"WARNING: {category_path} not found!")
            continue

        for folder_name in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            mapping = FRUIT_MAP.get(folder_name)
            if mapping is None:
                print(f"WARNING: Unknown folder '{folder_name}' — skipping")
                continue

            fruit_type, freshness = mapping

            # Collect image paths
            images = []
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(folder_path, f))

            # Cap to avoid imbalance
            if len(images) > MAX_IMAGES_PER_FOLDER:
                images = random.sample(images, MAX_IMAGES_PER_FOLDER)

            if freshness == "fresh":
                fresh_images.extend(images)
            else:
                rotten_images.extend(images)

            key = f"{freshness}_{fruit_type}"
            fruit_stats[key] = len(images)

    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"\n  Total fresh images (after cap): {len(fresh_images)}")
    print(f"  Total rotten images (after cap): {len(rotten_images)}")
    print(f"\n  Per-folder counts:")
    for key in sorted(fruit_stats.keys()):
        print(f"    {key}: {fruit_stats[key]}")

    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in ["fresh", "rotten"]:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

    # Split and copy
    for label, images in [("fresh", fresh_images), ("rotten", rotten_images)]:
        # 80% train, 10% val, 10% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest_dir = os.path.join(OUT_DIR, split_name, label)
            for i, img_path in enumerate(split_imgs):
                ext = Path(img_path).suffix
                dest_file = os.path.join(dest_dir, f"{label}_{i:05d}{ext}")
                shutil.copy2(img_path, dest_file)

        print(f"\n  {label}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Final verification
    print(f"\n{'=' * 60}")
    print("  FINAL DATASET COUNTS")
    print("=" * 60)
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["fresh", "rotten"]:
            p = os.path.join(OUT_DIR, split, cls)
            count = len(os.listdir(p)) if os.path.exists(p) else 0
            total += count
            print(f"  {split}/{cls}: {count} images")

    print(f"\n  TOTAL: {total} images")
    print(f"  Saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
