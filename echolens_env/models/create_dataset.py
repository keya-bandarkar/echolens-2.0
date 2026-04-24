import os
from pathlib import Path
import json

# Create dataset directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"

# Create folders
(DATASETS_PATH / "raw").mkdir(exist_ok=True)
(DATASETS_PATH / "processed").mkdir(exist_ok=True)
(DATASETS_PATH / "labels").mkdir(exist_ok=True)

print("📁 Dataset directories created!")

# Example gesture list
GESTURES = [
    "hello",
    "thank_you",
    "good_morning",
    "good_night",
    "yes",
    "no",
    "help",
    "please",
    "sorry",
    "love"
]

# Create gesture folders
for gesture in GESTURES:
    gesture_path = DATASETS_PATH / "raw" / gesture
    gesture_path.mkdir(exist_ok=True)
    print(f"✅ Created folder: {gesture}")

# Save gesture names
metadata = {
    "gestures": GESTURES,
    "total_gestures": len(GESTURES),
    "created": str(Path.cwd())
}

with open(DATASETS_PATH / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Dataset structure created with {len(GESTURES)} gestures!")
print(f"📍 Location: {DATASETS_PATH}")
print("\nNext step: Record videos for each gesture and place in:")
for gesture in GESTURES:
    print(f"  - {DATASETS_PATH / 'raw' / gesture}")