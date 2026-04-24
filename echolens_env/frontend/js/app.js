// Main Application Controller
class EchoLensApp {
    constructor() {
        this.state = {
            isRecording: false,
            sessionId: this.generateSessionId(),
            translationHistory: [],
            settings: {
                autoCorrect: true,
                saveHistory: true,
                soundEnabled: true
            }
        };
        
        this.apiClient = null;
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing EchoLens 2.0...');
        
        // Initialize API client
        this.apiClient = new APIClient('http://localhost:5000');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load settings
        this.loadSettings();
        
        // Check backend connection
        await this.checkBackendConnection();
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchPage(e.target.dataset.page));
        });
        
        // Translator controls
        document.getElementById('startBtn').addEventListener('click', () => this.startRecognition());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopRecognition());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetBuffer());
        document.getElementById('copyBtn').addEventListener('click', () => this.copyToClipboard());
        
        // Settings
        document.getElementById('autoCorrect').addEventListener('change', (e) => {
            this.state.settings.autoCorrect = e.target.checked;
            this.saveSettings();
        });
        
        document.getElementById('saveHistory').addEventListener('change', (e) => {
            this.state.settings.saveHistory = e.target.checked;
            this.saveSettings();
        });
        
        document.getElementById('soundEnabled').addEventListener('change', (e) => {
            this.state.settings.soundEnabled = e.target.checked;
            this.saveSettings();
        });
    }
    
    async startRecognition() {
        try {
            const video = document.getElementById('webcam');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            video.srcObject = stream;
            this.state.isRecording = true;
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            this.updateStatus('online');
            console.log('✅ Camera started');
            
            // Start frame processing loop
            this.processFramesLoop();
        } catch (error) {
            console.error('Error starting recognition:', error);
            this.showError('Failed to start camera. Check permissions.');
            this.updateStatus('offline');
        }
    }
    
    stopRecognition() {
        const video = document.getElementById('webcam');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        
        this.state.isRecording = false;
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        
        this.updateStatus('offline');
        console.log('⏹️ Camera stopped');
    }
    
    async processFramesLoop() {
        if (!this.state.isRecording) return;
        
        try {
            const canvas = document.getElementById('canvas');
            const video = document.getElementById('webcam');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw video frame
            ctx.drawImage(video, 0, 0);
            
            // Get frame as base64
            const frame = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to API
            try {
                const response = await this.apiClient.processFrame(frame, this.state.sessionId);
                this.updateResults(response);
            } catch (error) {
                console.error('Frame processing error:', error);
            }
            
            // Continue loop
            requestAnimationFrame(() => this.processFramesLoop());
        } catch (error) {
            console.error('Error in processing loop:', error);
        }
    }
    
    updateResults(response) {
        // Update buffer progress
        const bufferPercent = (response.buffer_size / 60) * 100;
        document.getElementById('bufferSize').textContent = response.buffer_size;
        document.getElementById('bufferProgress').style.width = bufferPercent + '%';
        
        // Update predictions
        if (response.prediction) {
            const pred = response.prediction;
            const displayText = pred.gesture;
            
            document.getElementById('recognizedText').innerHTML = `<p>${displayText}</p>`;
            document.getElementById('correctedText').innerHTML = `<p>${displayText}</p>`;
            
            // Update confidence
            const confidence = Math.round(pred.confidence * 100);
            document.getElementById('confidenceBar').style.width = confidence + '%';
            document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;
            
            // Enable copy button
            document.getElementById('copyBtn').disabled = false;
            
            // Add to history
            if (this.state.settings.saveHistory) {
                this.addToHistory(displayText, pred.confidence);
            }
            
            // Play sound if enabled
            if (this.state.settings.soundEnabled) {
                this.playNotificationSound();
            }
        }
    }
    
    resetBuffer() {
        this.apiClient.resetBuffer(this.state.sessionId);
        document.getElementById('bufferSize').textContent = '0';
        document.getElementById('bufferProgress').style.width = '0%';
        document.getElementById('recognizedText').innerHTML = '<p class="placeholder">👁️ Waiting for input...</p>';
        document.getElementById('correctedText').innerHTML = '<p class="placeholder">📝 Translation will appear here...</p>';
        document.getElementById('copyBtn').disabled = true;
    }
    
    switchPage(pageName) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
        
        // Show selected page
        document.getElementById(pageName).classList.add('active');
        
        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.page === pageName) {
                btn.classList.add('active');
            }
        });
    }
    
    copyToClipboard() {
        const text = document.getElementById('correctedText').innerText.replace('📝 Translation will appear here...', '');
        if (text.trim()) {
            navigator.clipboard.writeText(text).then(() => {
                this.showNotification('✅ Copied to clipboard!');
            });
        }
    }
    
    addToHistory(text, confidence) {
        const entry = {
            id: Date.now(),
            text,
            confidence,
            timestamp: new Date().toLocaleString()
        };
        
        this.state.translationHistory.unshift(entry);
        
        // Keep only last 50 entries
        if (this.state.translationHistory.length > 50) {
            this.state.translationHistory.pop();
        }
        
        this.updateHistoryDisplay();
        localStorage.setItem('translationHistory', JSON.stringify(this.state.translationHistory));
    }
    
    updateHistoryDisplay() {
        const historyList = document.getElementById('historyList');
        if (this.state.translationHistory.length === 0) {
            historyList.innerHTML = '<p class="placeholder">📋 No translations yet</p>';
            return;
        }
        
        historyList.innerHTML = this.state.translationHistory.map(entry => `
            <div class="history-item">
                <p class="history-text">${entry.text}</p>
                <span class="history-confidence">${Math.round(entry.confidence * 100)}%</span>
                <span class="history-time">${entry.timestamp}</span>
            </div>
        `).join('');
    }
    
    saveSettings() {
        localStorage.setItem('echolensSettings', JSON.stringify(this.state.settings));
    }
    
    loadSettings() {
        const saved = localStorage.getItem('echolensSettings');
        if (saved) {
            this.state.settings = JSON.parse(saved);
            document.getElementById('autoCorrect').checked = this.state.settings.autoCorrect;
            document.getElementById('saveHistory').checked = this.state.settings.saveHistory;
            document.getElementById('soundEnabled').checked = this.state.settings.soundEnabled;
        }
        
        // Load history
        const history = localStorage.getItem('translationHistory');
        if (history) {
            this.state.translationHistory = JSON.parse(history);
            this.updateHistoryDisplay();
        }
    }
    
    async checkBackendConnection() {
        try {
            const isHealthy = await this.apiClient.checkHealth();
            if (isHealthy) {
                this.updateStatus('online');
                console.log('✅ Connected to backend');
            } else {
                this.updateStatus('offline');
                console.log('❌ Backend not responding');
            }
        } catch (error) {
            this.updateStatus('offline');
            console.error('Backend connection failed:', error);
        }
    }
    
    updateStatus(status) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        
        if (status === 'online') {
            indicator.classList.remove('offline');
            indicator.classList.add('online');
            text.textContent = '✅ Connected';
        } else {
            indicator.classList.remove('online');
            indicator.classList.add('offline');
            text.textContent = '❌ Disconnected';
        }
    }
    
    showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 9999;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    showError(message) {
        const error = document.createElement('div');
        error.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 9999;
        `;
        error.textContent = message;
        document.body.appendChild(error);
        
        setTimeout(() => error.remove(), 5000);
    }
    
    playNotificationSound() {
        // Create a simple beep sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();
        
        oscillator.connect(gain);
        gain.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gain.gain.setValueAtTime(0.3, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Add slide animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new EchoLensApp();
    console.log('🎉 EchoLens 2.0 Frontend Ready!');
});