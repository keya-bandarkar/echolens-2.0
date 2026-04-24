class APIClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async processFrame(frameData, sessionId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/process-frame`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame: frameData,
                    session_id: sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
    
    async resetBuffer(sessionId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/reset-buffer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }
}