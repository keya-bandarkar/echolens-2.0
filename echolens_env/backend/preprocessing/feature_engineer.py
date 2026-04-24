import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Convert raw keypoints to engineered features."""
    
    def __init__(self, feature_dim: int = 100):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        logger.info(f"FeatureEngineer initialized with dimension {feature_dim}")
    
    def engineer_features(self, keypoint_sequence: List[Dict]) -> np.ndarray:
        """
        Convert raw keypoints to engineered features.
        
        Args:
            keypoint_sequence: List of keypoint dictionaries from frames
            
        Returns:
            Numpy array of shape (num_frames, feature_dim)
        """
        features = []
        
        for frame_data in keypoint_sequence:
            frame_features = self._extract_frame_features(frame_data)
            features.append(frame_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_frame_features(self, frame_data: Dict) -> np.ndarray:
        """Extract features from single frame."""
        features = []
        
        # Pose features (take key joints)
        pose = frame_data['pose']
        if pose is not None and len(pose) > 0:
            # Use first 48 values (12 key joints * 4 values)
            pose_features = pose[:48]
            features.extend(pose_features)
        else:
            features.extend([0] * 48)
        
        # Hand features (both hands if present)
        hands = frame_data['hands']
        for i in range(2):  # Support 2 hands
            if i < len(hands):
                hand_features = hands[i][:21]  # 21 keypoints per hand
                features.extend(hand_features)
            else:
                features.extend([0] * 21)
        
        # Pad or truncate to exact feature dimension
        features = features[:self.feature_dim]
        if len(features) < self.feature_dim:
            features.extend([0] * (self.feature_dim - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def compute_motion_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute motion/velocity features from keypoint sequence.
        
        Args:
            sequence: Shape (num_frames, feature_dim)
            
        Returns:
            Motion features (velocity between frames)
        """
        if len(sequence) < 2:
            return sequence
        
        motion = np.diff(sequence, axis=0)  # Frame-to-frame differences
        
        # Pad first frame to match original length
        motion = np.vstack([motion[0], motion])
        
        return motion
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean, unit variance.
        
        Args:
            features: Shape (num_frames, feature_dim)
            
        Returns:
            Normalized features
        """
        original_shape = features.shape
        reshaped = features.reshape(-1, features.shape[-1])
        normalized = self.scaler.fit_transform(reshaped)
        return normalized.reshape(original_shape)
    
    def augment_features(self, features: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add slight noise for data augmentation.
        
        Args:
            features: Input features
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Augmented features
        """
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise