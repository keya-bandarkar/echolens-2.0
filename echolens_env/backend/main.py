from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import logging
from pathlib import Path
import base64
import cv2
from io import BytesIO

# Import our modules
from preprocessing.frame_processor import FrameProcessor
#from preprocessing.keypoint_extractor import KeypointExtractor
from preprocessing.feature_engineer import FeatureEngineer
from models.lstm_model import LSTMSignLanguageModel
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
logger.info("Initializing EchoLens 2.0 Backend...")

frame_processor = FrameProcessor(
    target_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT)
)
#keypoint_extractor = KeypointExtractor()
feature_engineer = FeatureEngineer()

# Load model
logger.info("Loading model...")
model = LSTMSignLanguageModel(
    input_shape=(config.SEQUENCE_LENGTH, 100),
    num_classes=config.NUM_CLASSES
)
model.build_model()
model.compile_model()

# Load pretrained weights if they exist
model_path = config.MODELS_PATH / "trained_models" / "lstm_model.h5"
if model_path.exists():
    model.load_model(str(model_path))
    logger.info(f"✅ Model loaded from {model_path}")
else:
    logger.warning(f"⚠️ No pretrained model found at {model_path}")
    logger.warning("Using untrained model - train first!")

# Load gesture names
gesture_names_path = config.MODELS_PATH / "trained_models" / "gesture_names.json"
if gesture_names_path.exists():
    with open(gesture_names_path, 'r') as f:
        gesture_names = json.load(f)
else:
    # Default gesture names (update after training)
    gesture_names = [f"gesture_{i}" for i in range(config.NUM_CLASSES)]
    logger.warning(f"Using default gesture names")

# Sequence buffer for real-time processing
class InferenceBuffer:
    """Buffer for real-time sequence inference."""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add_features(self, features):
        """Add features to buffer."""
        if len(self.buffer) >= self.sequence_length:
            self.buffer.pop(0)
        self.buffer.append(features)
    
    def get_sequence(self):
        """Get sequence if complete."""
        if len(self.buffer) == self.sequence_length:
            return np.array(self.buffer)
        return None
    
    def clear(self):
        """Clear buffer."""
        self.buffer = []
    
    def get_progress(self):
        """Get buffer fill percentage."""
        return len(self.buffer) / self.sequence_length

# Global buffers for different sessions
inference_buffers = {}

# ROUTES

@app.route('/')
def index():
    """Health check."""
    return jsonify({
        'status': 'online',
        'app': 'EchoLens 2.0',
        'version': '1.0',
        'num_classes': config.NUM_CLASSES,
        'sequence_length': config.SEQUENCE_LENGTH
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check."""
    return jsonify({'status': 'healthy'})

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """
    Process a single frame and update inference buffer.
    
    Expected JSON:
    {
        'frame': 'base64_encoded_image',
        'session_id': 'unique_session_id'
    }
    """
    try:
        data = request.json
        frame_base64 = data.get('frame')
        session_id = data.get('session_id', 'default')
        
        if not frame_base64:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode frame
        frame_data = base64.b64decode(frame_base64.split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract keypoints
        keypoints = keypoint_extractor.extract_keypoints(frame)
        
        # Engineer features
        features = feature_engineer.engineer_features([keypoints])[0]
        
        # Add to buffer
        if session_id not in inference_buffers:
            inference_buffers[session_id] = InferenceBuffer(config.SEQUENCE_LENGTH)
        
        inference_buffers[session_id].add_features(features)
        
        # Prepare response
        result = {
            'status': 'frame_processed',
            'buffer_progress': inference_buffers[session_id].get_progress(),
            'buffer_size': len(inference_buffers[session_id].buffer),
            'prediction': None
        }
        
        # Try to predict if buffer is full
        sequence = inference_buffers[session_id].get_sequence()
        if sequence is not None:
            # Make prediction
            prediction = model.model.predict(np.array([sequence]), verbose=0)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            result['prediction'] = {
                'gesture': gesture_names[class_idx],
                'confidence': confidence,
                'class_id': int(class_idx)
            }
            
            logger.info(f"Prediction: {gesture_names[class_idx]} ({confidence:.2f})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/reset-buffer', methods=['POST'])
def reset_buffer():
    """Reset inference buffer for a session."""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in inference_buffers:
            inference_buffers[session_id].clear()
        
        return jsonify({'status': 'buffer_reset'})
    
    except Exception as e:
        logger.error(f"Error resetting buffer: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Get list of all recognized gestures."""
    return jsonify({
        'count': len(gesture_names),
        'gestures': gesture_names
    })

# Error handlers

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

# Main

if __name__ == '__main__':
    logger.info("🚀 Starting EchoLens 2.0 Backend Server...")
    logger.info(f"📊 Loaded {len(gesture_names)} gesture classes")
    logger.info(f"🎯 Sequence length: {config.SEQUENCE_LENGTH} frames")
    logger.info("🌐 Server running on http://localhost:5000")
    app.run(debug=True, port=5000)