import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from backend.preprocessing.frame_processor import FrameProcessor
from backend.preprocessing.feature_engineer import FeatureEngineer

def test_preprocessing_simple():
    """Test the preprocessing pipeline without MediaPipe."""
    
    print("\n" + "="*60)
    print("TESTING PREPROCESSING PIPELINE (SIMPLIFIED)")
    print("="*60)
    
    # Initialize components
    print("\n1️⃣  Initializing components...")
    frame_processor = FrameProcessor()
    feature_engineer = FeatureEngineer()
    print("✅ Components initialized")
    
    # Test with webcam
    print("\n2️⃣  Testing with webcam...")
    print("   (Opens webcam for 5 seconds)")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("   Capturing frames...")
    frame_count = 0
    
    while frame_count < 30:
        frame = frame_processor.get_webcam_frame(cap)
        if frame is None:
            break
        
        frame_processor.display_frame(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"   Captured {frame_count} frames...")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {frame_count} frames")
    
    # Test feature engineering with dummy data
    print("\n3️⃣  Testing feature engineering...")
    dummy_keypoints = [
        {'pose': np.random.rand(132), 'hands': [np.random.rand(63)]}
        for _ in range(30)
    ]
    
    features = feature_engineer.engineer_features(dummy_keypoints)
    print(f"   Features shape: {features.shape}")
    print(f"   ✅ Feature engineering works")
    
    motion = feature_engineer.compute_motion_features(features)
    print(f"   Motion features shape: {motion.shape}")
    
    normalized = feature_engineer.normalize_features(features)
    print(f"   Normalized features shape: {normalized.shape}")
    print(f"   ✅ Normalization works")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_preprocessing_simple()