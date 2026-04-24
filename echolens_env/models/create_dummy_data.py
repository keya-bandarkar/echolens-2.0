import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"

# Load metadata
with open(DATASETS_PATH / "metadata.json", "r") as f:
    metadata = json.load(f)

gestures = metadata["gestures"]

# Create dummy processed data
print("📊 Creating dummy gesture data for testing...")

processed_path = DATASETS_PATH / "processed"

for gesture_idx, gesture in enumerate(gestures):
    gesture_path = processed_path / gesture
    gesture_path.mkdir(exist_ok=True)
    
    # Create 20 dummy samples per gesture
    for sample_idx in range(20):
        # Generate random keypoint sequence
        # Shape: (60 frames, 100 features)
        features = np.random.rand(60, 100).astype(np.float32)
        
        # Save as numpy file
        file_path = gesture_path / f"sample_{sample_idx:02d}.npy"
        np.save(file_path, features)
    
    print(f"✅ Created 20 samples for: {gesture}")

print(f"\n✅ Dummy data created in {processed_path}")
print("Ready to train!")