import sys
from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.lstm_model import LSTMSignLanguageModel
from backend.preprocessing.feature_engineer import FeatureEngineer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"
MODELS_PATH = PROJECT_ROOT / "models"

def load_dataset(processed_data_path):
    """Load preprocessed gesture dataset."""
    logger.info("Loading dataset...")
    
    X = []
    y = []
    gesture_names = []
    
    processed_path = Path(processed_data_path)
    
    # Get gesture folders
    gesture_folders = sorted([d for d in processed_path.iterdir() if d.is_dir()])
    
    if not gesture_folders:
        logger.error(f"No gesture folders found in {processed_data_path}")
        return None, None, None
    
    gesture_names = [d.name for d in gesture_folders]
    logger.info(f"Found {len(gesture_names)} gestures: {gesture_names}")
    
    for gesture_idx, gesture_path in enumerate(gesture_folders):
        logger.info(f"Loading gesture: {gesture_path.name}")
        
        # Load all .npy files for this gesture
        for feature_file in gesture_path.glob("*.npy"):
            try:
                features = np.load(feature_file)
                
                # Ensure correct shape (60, 100)
                if features.shape[0] < 60:
                    # Pad if too short
                    pad_size = 60 - features.shape[0]
                    features = np.pad(features, ((0, pad_size), (0, 0)))
                else:
                    # Truncate if too long
                    features = features[:60]
                
                X.append(features)
                y.append(gesture_idx)
                
            except Exception as e:
                logger.warning(f"Error loading {feature_file}: {e}")
    
    if not X:
        logger.error("No data loaded!")
        return None, None, None
    
    X = np.array(X)
    y = np.eye(len(gesture_names))[y]  # One-hot encoding
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    
    return X, y, gesture_names

def train_lstm_model():
    """Train LSTM gesture recognition model."""
    
    logger.info("="*60)
    logger.info("TRAINING LSTM GESTURE RECOGNITION MODEL")
    logger.info("="*60)
    
    # Load dataset
    X, y, gesture_names = load_dataset(DATASETS_PATH / "processed")
    
    if X is None:
        logger.error("Failed to load dataset!")
        return False
    
    num_gestures = len(gesture_names)
    
    # Split dataset
    logger.info("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Create and train model
    logger.info("\nBuilding model...")
    model = LSTMSignLanguageModel(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_gestures
    )
    model.build_model()
    model.compile_model()
    
    logger.info(f"Model built with {num_gestures} gesture classes")
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*60)
    
    test_loss, test_acc = model.model.evaluate(X_test, y_test)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    logger.info("\nSaving model...")
    trained_models_path = MODELS_PATH / "trained_models"
    trained_models_path.mkdir(exist_ok=True)
    
    model.save_model(str(trained_models_path / "lstm_model.h5"))
    
    # Save gesture names
    with open(trained_models_path / "gesture_names.json", "w") as f:
        json.dump(gesture_names, f, indent=2)
    
    logger.info(f"✅ Model saved to {trained_models_path}")
    logger.info(f"✅ Gesture names saved")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE! 🎉")
    logger.info("="*60)
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Model ready at: {trained_models_path / 'lstm_model.h5'}")
    
    return True

if __name__ == "__main__":
    success = train_lstm_model()
    if not success:
        logger.error("Training failed!")
        exit(1)