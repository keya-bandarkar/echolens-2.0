import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATASETS_PATH = PROJECT_ROOT / "datasets"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"

# Video processing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
SEQUENCE_LENGTH = 60  # Number of frames for gesture

# Model
NUM_CLASSES = 250  # Adjust based on your gestures
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# MediaPipe
POSE_CONFIDENCE = 0.5
HAND_CONFIDENCE = 0.5

print("✅ Configuration loaded")


