from datasets import load_dataset
from PIL import Image
import os

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_NAME = "Parveshiiii/AI-vs-Real"

# Total per class = 4000
TRAIN_COUNT = 2800
VAL_COUNT = 600
TEST_COUNT = 600

# -----------------------------
# CREATE FOLDERS
# -----------------------------

folders = [
    "dataset/train/AI",
    "dataset/train/REAL",
    "dataset/validation/AI",
    "dataset/validation/REAL",
    "dataset/test/AI",
    "dataset/test/REAL"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------

print("Connecting to Hugging Face...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

# Shuffle stream so images are not taken in original order
ds = ds.shuffle(seed=42, buffer_size=1000)

# -----------------------------
# COUNTERS
# -----------------------------

counts = {
    0: 0,
    1: 0
}

saved = {
    "AI": {
        "train": 0,
        "validation": 0,
        "test": 0
    },
    "REAL": {
        "train": 0,
        "validation": 0,
        "test": 0
    }
}

# -----------------------------
# SAVE FUNCTION
# -----------------------------

def save_image(image, label, index):

    if label == 0:
        class_name = "AI"
    else:
        class_name = "REAL"

    total = saved[class_name]["train"] + \
            saved[class_name]["validation"] + \
            saved[class_name]["test"]

    if total < TRAIN_COUNT:
        split = "train"
    elif total < TRAIN_COUNT + VAL_COUNT:
        split = "validation"
    elif total < TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        split = "test"
    else:
        return False

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((128, 128))

    filename = f"{class_name.lower()}_{index:05d}.jpg"

    path = os.path.join(
        "dataset",
        split,
        class_name,
        filename
    )

    image.save(
        path,
        "JPEG",
        quality=90
    )

    saved[class_name][split] += 1

    return True


# -----------------------------
# EXTRACT IMAGES
# -----------------------------

print("Starting extraction...")
print("Please wait...")
print()

for item in ds:

    label = int(item["binary_label"])

    # We only need labels 0 and 1
    if label not in [0, 1]:
        continue

    # Stop when both classes reach 4000
    if counts[0] >= 4000 and counts[1] >= 4000:
        break

    # Skip if this class already has 4000
    if counts[label] >= 4000:
        continue

    image = item["image"]

    class_name = "AI" if label == 0 else "REAL"

    index = counts[label]

    save_image(
        image,
        label,
        index
    )

    counts[label] += 1

    total_saved = counts[0] + counts[1]

    if total_saved % 100 == 0:
        print(
            f"Progress: AI={counts[0]}/4000 | "
            f"REAL={counts[1]}/4000"
        )

# -----------------------------
# FINAL RESULT
# -----------------------------

print()
print("================================")
print("EXTRACTION COMPLETE")
print("================================")

print("AI:")
print("  Train:", saved["AI"]["train"])
print("  Validation:", saved["AI"]["validation"])
print("  Test:", saved["AI"]["test"])

print()
print("REAL:")
print("  Train:", saved["REAL"]["train"])
print("  Validation:", saved["REAL"]["validation"])
print("  Test:", saved["REAL"]["test"])

print()
print("Dataset ready!")