/**
 * Main ChessGame class handling game state and UI coordination
 */
class ChessGame {
    constructor() {
        this.currentFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
        this.selectedModels = [];
        this.gameHistory = [];
        this.selectedSquare = null;
        this.isAnalyzing = false;
        this.websocket = null;
        this.currentAnalysis = {
            stockfish_eval: null,
            model_responses: [],
            debate_rounds: [],
            phase: 'idle'
        };
        
        // NEW: Persistent storage for all analyses and moves
        this.analysisHistory = []; // Stores all previous analyses
        this.moveCounter = 0; // Track move numbers
        
        this.init();
    }

    async init() {
        await this.loadModels();
        this.renderBoard();
        this.updateGameStatus();
        this.connectWebSocket();
        // Always get Stockfish evaluation for the current position
        await this.updateStockfishEvaluation();
    }

    async updateStockfishEvaluation() {
        try {
            const response = await fetch('/api/stockfish_eval');
            if (response.ok) {
                this.currentAnalysis.stockfish_eval = await response.json();
                await this.updateAnalysisDisplay();
            }
        } catch (error) {
            console.error('Error updating Stockfish evaluation:', error);
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected, attempting to reconnect...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        console.log('WebSocket message:', data);
        
        switch (data.type) {
            case 'analysis_start':
                this.currentAnalysis = {
                    stockfish_eval: null,
                    model_responses: [],
                    debate_rounds: [],
                    phase: 'starting'
                };
                this.updateAnalysisDisplay();
                break;
                
            case 'stockfish_complete':
                this.currentAnalysis.stockfish_eval = data.stockfish_evaluation;
                this.updateAnalysisDisplay();
                break;
                
            case 'phase_start':
                this.currentAnalysis.phase = data.phase;
                this.updateAnalysisDisplay();
                break;
                
            case 'model_start':
                this.showModelProgress(data);
                break;
                
            case 'model_complete':
                this.addModelResponse(data);
                this.updateAnalysisDisplay();
                break;
                
            case 'conflict_detected':
                this.currentAnalysis.phase = 'conflict';
                this.updateAnalysisDisplay();
                break;
                
            case 'analysis_complete':
                this.currentAnalysis.phase = 'complete';
                this.currentAnalysis.final_result = data.final_result;
                this.updateAnalysisDisplay();
                break;
        }
    }

    addModelResponse(data) {
        // Add or update model response
        const existingIndex = this.currentAnalysis.model_responses.findIndex(
            r => r.model_name === data.model_name && r.phase === data.phase
        );
        
        const responseData = {
            ...data.response,
            phase: data.phase,
            model_index: data.model_index
        };
        
        if (existingIndex >= 0) {
            this.currentAnalysis.model_responses[existingIndex] = responseData;
        } else {
            this.currentAnalysis.model_responses.push(responseData);
        }
    }

    showModelProgress(data) {
        // Show that a model is currently working
        document.getElementById('analysisContent').innerHTML += `
            <div id="progress-${data.model_name}-${data.phase}" class="loading">
                <div class="spinner"></div>
                ${data.message}
            </div>
        `;
    }

    saveAnalysisToHistory() {
        if (this.currentAnalysis.final_result && this.currentAnalysis.final_result.final_move) {
            this.moveCounter++;
            const analysisSnapshot = {
                moveNumber: this.moveCounter,
                position: this.currentFen,
                timestamp: new Date().toLocaleTimeString(),
                selectedMove: this.currentAnalysis.final_result.final_move,
                consensusStrength: this.currentAnalysis.final_result.consensus_strength,
                hadDebate: this.currentAnalysis.final_result.had_debate,
                stockfish_eval: JSON.parse(JSON.stringify(this.currentAnalysis.stockfish_eval)),
                model_responses: JSON.parse(JSON.stringify(this.currentAnalysis.model_responses)),
                debate_decision: this.currentAnalysis.final_result.debate_decision
            };
            
            this.analysisHistory.push(analysisSnapshot);
            console.log(`Saved analysis #${this.moveCounter} to history`, analysisSnapshot);
        }
    }

    async resetGame() {
        try {
            const response = await fetch('/api/reset_game', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                this.currentFen = data.fen;
                this.gameHistory = [];
                
                // Reset current analysis but KEEP analysis history
                this.currentAnalysis = {
                    stockfish_eval: null,
                    model_responses: [],
                    debate_rounds: [],
                    phase: 'idle'
                };
                
                this.renderBoard();
                this.updateGameStatus();
                
                // Get Stockfish evaluation for the starting position
                await this.updateStockfishEvaluation();
                
                // Update display to show history + new position
                await this.updateAnalysisDisplay();
            }
        } catch (error) {
            console.error('Error resetting game:', error);
        }
    }
}