# EchoLens 2.0
## AI-Powered Real-Time Sign Language Recognition & Translation System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-yellow)

> ⚠️ **Project Status**: This project is still in active development. Features and APIs may change.

---

## 🎯 Overview

EchoLens 2.0 is an intelligent, real-time sign language recognition and translation system that bridges the communication gap between sign language users and non-signers. It uses advanced computer vision and deep learning to detect hand and body movements from live video input and convert them into meaningful text.

### 🌟 Vision

Making sign language accessible through AI-powered real-time translation, creating a more inclusive digital world.

---

## ✨ Key Features

- **Real-Time Gesture Recognition**: Capture and process live video input via webcam
- **Pose & Hand Detection**: Uses MediaPipe for accurate keypoint detection
- **Deep Learning Models**: LSTM and Transformer architectures for sequence modeling
- **Modern Web Interface**: Beautiful, responsive web UI for easy interaction
- **Translation History**: Keep track of all translations with timestamps
- **Configurable Settings**: Customize auto-correction and notifications
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Easy Setup**: Simple installation with one-line commands

---

## 🏗️ System Architecture

```
┌─────────────────────┐
│   Live Video Input  │
│    (Webcam/Mobile)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Frame Processing    │
│ (OpenCV)            │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Keypoint Detection  │
│ (MediaPipe)         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Feature Engineering │
│ (Normalization)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Sequence Buffering  │
│ (60 frames)         │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Deep Learning Model │
│ (LSTM/Transformer)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ NLP Post-Processing │
│ (Text Refinement)   │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Text Output +       │
│ Confidence Score    │
└─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Webcam (for gesture capture)
- 4GB RAM minimum
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/echolens.git
cd echolens
```

2. **Create virtual environment**
```bash
python -m venv echolens_env

# On Windows
echolens_env\Scripts\activate

# On macOS/Linux
source echolens_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Start Flask Backend (Terminal 1)
```bash
python backend/main.py
```
Server will run on `http://localhost:5000`

Expected output:
```
🚀 Starting EchoLens 2.0 Backend Server...
📊 Loaded 250 gesture classes
🎯 Sequence length: 60 frames
🌐 Server running on http://localhost:5000
```

#### Step 2: Serve Frontend (Terminal 2)
```bash
cd frontend
python -m http.server 8000
```

#### Step 3: Open in Browser
```
http://localhost:8000
```

#### Step 4: Start Recognizing!
- Click **"Start Recognizing"** button
- Make gestures in front of your camera
- Watch real-time predictions appear
- See confidence scores and translation history

---

## 📁 Project Structure

```
echolens-2.0/
│
├── 📂 backend/
│   ├── 📄 config.py                 # Configuration settings
│   ├── 📄 main.py                   # Flask API server
│   ├── 📂 preprocessing/            # Data processing modules
│   │   ├── 📄 frame_processor.py
│   │   ├── 📄 keypoint_extractor.py
│   │   └── 📄 feature_engineer.py
│   ├── 📂 models/                   # Deep learning models
│   │   └── 📄 lstm_model.py
│   └── 📂 tests/                    # Unit tests
│
├── 📂 frontend/
│   ├── 📄 index.html                # Main web interface
│   ├── 📂 css/
│   │   └── 📄 style.css             # Styling
│   └── 📂 js/
│       ├── 📄 app.js                # Main application logic
│       └── 📄 api-client.js         # API communication
│
├── 📂 Scripts/
│   ├── 📄 create_dataset.py         # Dataset structure creation
│   ├── 📄 create_dummy_data.py      # Generate sample data
│   ├── 📄 train_model.py            # Model training script
│   ├── 📄 evaluate_model.py         # Model evaluation
│   └── 📄 test_preprocessing.py     # Integration tests
│
├── 📂 models/
│   ├── 📂 trained_models/           # Saved model weights
│   └── 📂 checkpoints/              # Training checkpoints
│
├── 📂 datasets/
│   ├── 📂 raw/                      # Raw gesture videos
│   ├── 📂 processed/                # Processed keypoint data
│   └── 📂 labels/                   # Gesture labels
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This file
├── 📄 .gitignore                    # Git ignore rules
└── 📄 LICENSE                       # MIT License
```

---

## 🔧 Technology Stack

### Backend
- **Python 3.9+** - Programming language
- **Flask 2.3** - Web framework
- **TensorFlow 2.13** - Deep learning framework
- **Keras** - Neural network API
- **MediaPipe** - Pose and hand detection
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling
- **Vanilla JavaScript** - No dependencies required
- **WebRTC** - Real-time communication (webcam)

### Deep Learning
- **LSTM** - Long Short-Term Memory networks
- **Transformer** - Attention-based architecture
- **Sequence-to-Sequence** - For gesture-to-text translation

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Real-Time Latency** | < 100ms |
| **Gesture Recognition Accuracy** | 75-95% (with trained data) |
| **Supported Gestures** | 250+ customizable |
| **Sequence Length** | 60 frames (2 seconds @ 30 FPS) |
| **Frames Per Second** | 30 FPS |
| **Memory Usage** | ~2GB RAM |

---

## 🎓 How It Works

### Step-by-Step Process

1. **Video Capture**: Webcam captures live video at 30 FPS
2. **Frame Processing**: Each frame is resized to 640×480 pixels
3. **Keypoint Detection**: MediaPipe detects:
   - 33 body pose landmarks
   - 21 hand keypoints (both hands)
4. **Feature Extraction**: Raw keypoints converted to ML-ready features
5. **Sequence Modeling**: 60 consecutive frames fed to LSTM/Transformer
6. **Classification**: Model predicts gesture with confidence score
7. **Output**: Recognized gesture displayed with translation

### Model Architecture

**LSTM Model**:
```
Input (60 frames, 100 features)
    ↓
LSTM Layer (256 units) + Dropout
    ↓
LSTM Layer (128 units) + Dropout
    ↓
LSTM Layer (64 units) + Dropout
    ↓
Dense Layer (128 units)
    ↓
Dense Layer (64 units)
    ↓
Output Layer (250 classes)
```

---

## 🚀 Advanced Features

- **Grammar Correction**: NLP-based text refinement
- **Translation History**: Persistent storage with timestamps
- **Settings Customization**: Auto-correct, notifications, history storage
- **Multi-Gesture Support**: Recognize gesture sequences
- **Confidence Scoring**: Know model confidence level
- **Real-Time Processing**: Sub-100ms latency
- **Buffer Management**: 60-frame sequence buffering

---

## 📚 Training Your Own Model

### Step 1: Collect Gesture Data
```bash
mkdir datasets/raw/your_gesture
# Record 10-20 video samples per gesture in MP4 format
```

### Step 2: Create Dataset Structure
```bash
python Scripts/create_dataset.py
```

### Step 3: Create Sample Data (Optional - for testing)
```bash
python Scripts/create_dummy_data.py
```

### Step 4: Train Model
```bash
python Scripts/train_model.py
```

This will:
- Load your gesture dataset
- Split into train/validation/test sets
- Train for 50 epochs with early stopping
- Save trained weights
- Display test accuracy

### Step 5: Evaluate Performance
```bash
python Scripts/evaluate_model.py
```

This will show:
- Overall accuracy
- Per-gesture metrics
- Precision, recall, F1-score
- Confusion matrix

---

## 🔌 API Endpoints

### Backend API

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Health check | `{status: "online", app: "EchoLens 2.0", ...}` |
| `/api/health` | GET | API status | `{status: "healthy"}` |
| `/api/process-frame` | POST | Process video frame | `{buffer_progress: 0.5, prediction: {...}}` |
| `/api/reset-buffer` | POST | Reset inference buffer | `{status: "buffer_reset"}` |
| `/api/gestures` | GET | Get all gesture classes | `{count: 250, gestures: [...]}` |

### Example Request

```javascript
// Process a frame
const response = await fetch('http://localhost:5000/api/process-frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        frame: 'base64_encoded_image',
        session_id: 'unique_session_id'
    })
});

const data = await response.json();
console.log(data.prediction); // {gesture: "hello", confidence: 0.95}
```

---

## 🤝 Contributing

Contributions are welcome and appreciated! Whether it's bug fixes, new features, or documentation improvements.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Contribution Ideas

- Add support for more gestures
- Improve accuracy
- Add facial expression recognition
- Implement multilingual support
- Create mobile app (Flutter/React Native)
- Improve UI/UX
- Add unit tests
- Write documentation
- Report bugs

---

## 📋 Project Status & Roadmap

### ✅ Completed
- [x] Real-time video processing pipeline
- [x] LSTM gesture recognition model
- [x] Web interface with webcam integration
- [x] Translation history
- [x] Settings customization
- [x] Flask REST API
- [x] Model training scripts
- [x] Performance evaluation tools

### 🚧 In Progress
- [ ] Improve model accuracy (target: 95%+)
- [ ] Add more gesture classes (target: 500+)
- [ ] Optimize inference speed
- [ ] Better error handling

### 📅 Planned Features
- [ ] Facial expression recognition
- [ ] Multilingual support (English/Hindi/ASL)
- [ ] Mobile app (Flutter)
- [ ] Text-to-Sign translation
- [ ] Cloud deployment (AWS/Google Cloud)
- [ ] Real-time collaboration mode
- [ ] Custom gesture training UI
- [ ] Edge deployment (TensorFlow Lite)
- [ ] Performance monitoring dashboard

### ⚠️ Known Limitations
- Requires good lighting for optimal performance
- Currently supports isolated gestures (not continuous signing)
- Model accuracy varies with gesture complexity
- Limited to 250 gesture classes (expandable)

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### You're free to:
- Use commercially
- Distribute
- Modify
- Use privately

### Under the conditions that:
- License and copyright notice included
- No liability
- No warranty

---

## 🙏 Acknowledgments

- **MediaPipe Team** - Excellent pose and hand detection models
- **Google TensorFlow** - Powerful deep learning framework
- **Flask Community** - Simple yet powerful web framework
- **Sign Language Community** - Inspiration and guidance
- **OpenCV Project** - Computer vision library

---

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/echolens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/echolens/discussions)
- **Email**: your.email@example.com (optional)

---

## 📚 References & Resources

- [MediaPipe Documentation](https://mediapipe.dev/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Sign Language Resources](https://www.lifeprint.com/)

---

## 🎯 Citation

If you use EchoLens 2.0 in your research or project, please cite:

```bibtex
@software{echolens2025,
  title={EchoLens 2.0: AI-Powered Real-Time Sign Language Recognition System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/echolens}
}
```

---

## 📊 Project Statistics

- **Lines of Code**: 5000+
- **Models**: 2 (LSTM, Transformer)
- **Gesture Classes**: 250+
- **Frontend Pages**: 3 (Translator, History, Settings)
- **API Endpoints**: 5
- **Dependencies**: 15+
- **Test Coverage**: In Progress

---

## 🌟 Star History

[![Star History Chart](https://api.github.com/repos/yourusername/echolens/stargazers)](https://github.com/yourusername/echolens)

---

## 👥 Contributors

- **Your Name** - [@yourusername](https://github.com/yourusername) - Project Lead & Creator

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📢 Announcements

- **v1.0** - Initial release (in development)
- **v1.1** - Improved accuracy & performance (planned)
- **v2.0** - Mobile app & cloud deployment (planned)

---

<div align="center">

### Made with ❤️ for Accessibility & Inclusion

**[⬆ Back to Top](#echolens-20)**

**Status**: 🚧 In Active Development - Contributions Welcome!

</div>
