import mediapipe as mp
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeypointExtractor:
    """Extract pose and hand keypoints using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✅ KeypointExtractor initialized")
    
    def extract_keypoints(self, frame: np.ndarray) -> Dict:
        """
        Extract pose and hand keypoints from frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with pose and hand landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Detect pose
        pose_results = self.pose.process(frame_rgb)
        pose_landmarks = self._parse_pose_landmarks(pose_results)
        
        # Detect hands
        hand_results = self.hands.process(frame_rgb)
        hand_landmarks = self._parse_hand_landmarks(hand_results)
        
        return {
            'pose': pose_landmarks,
            'hands': hand_landmarks,
            'timestamp': time.time(),
            'pose_detected': pose_results.pose_landmarks is not None,
            'hands_detected': len(hand_landmarks) > 0
        }
    
    def _parse_pose_landmarks(self, results) -> Optional[np.ndarray]:
        """Convert MediaPipe pose landmarks to numpy array."""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(landmarks, dtype=np.float32)
        return np.zeros(33 * 4, dtype=np.float32)  # 33 keypoints * 4 values
    
    def _parse_hand_landmarks(self, results) -> List[np.ndarray]:
        """Convert MediaPipe hand landmarks to array."""
        hand_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                hand_data.append(np.array(landmarks, dtype=np.float32))
        return hand_data
    
    def close(self):
        """Release resources."""
        self.pose.close()
        self.hands.close()
        logger.info("KeypointExtractor closed")