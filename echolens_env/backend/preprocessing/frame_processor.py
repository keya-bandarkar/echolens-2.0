import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Handle video frame extraction and preprocessing."""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        self.target_size = target_size
        logger.info(f"FrameProcessor initialized with target size {target_size}")
    
    def extract_frames_from_video(self, video_path: str, fps: int = 30) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            
        Returns:
            List of frame arrays
        """
        logger.info(f"Extracting frames from {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        frames = []
        frame_count = 0
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, video_fps // fps)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified fps
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Histogram equalization for better feature extraction
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        return equalized / 255.0
    
    def get_webcam_frame(self, cap: cv2.VideoCapture) -> np.ndarray:
        """Capture frame from webcam."""
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, self.target_size)
            return frame
        return None
    
    @staticmethod
    def display_frame(frame: np.ndarray, window_name: str = "Frame"):
        """Display frame (for testing)."""
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str):
        """Save frame to file."""
        cv2.imwrite(output_path, frame)