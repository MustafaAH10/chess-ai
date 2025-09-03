/**
 * Move engine and game logic functionality
 */
ChessGame.prototype.loadModels = async function() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        this.renderModelSelection(data.models);
    } catch (error) {
        console.error('Error loading models:', error);
    }
};

ChessGame.prototype.renderModelSelection = function(models) {
    const container = document.getElementById('modelSelection');
    container.innerHTML = '';

    models.forEach(model => {
        const card = document.createElement('button');
        card.className = 'model-card';
        card.textContent = this.getModelDisplayName(model);
        card.onclick = () => this.toggleModel(model, card);
        container.appendChild(card);
    });
};

ChessGame.prototype.toggleModel = function(model, card) {
    if (this.selectedModels.includes(model)) {
        this.selectedModels = this.selectedModels.filter(m => m !== model);
        card.classList.remove('selected');
    } else {
        this.selectedModels.push(model);
        card.classList.add('selected');
    }
    
    // Update analyze button state
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = this.selectedModels.length === 0 || this.isAnalyzing;
};

ChessGame.prototype.updateSelectedModels = function() {
    const checkboxes = document.querySelectorAll('#modelList input[type="checkbox"]:checked');
    this.selectedModels = Array.from(checkboxes).map(cb => cb.value);
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = this.selectedModels.length === 0 || this.isAnalyzing;
    
    // Update display showing selected count
    const selectedCount = document.getElementById('selectedCount');
    if (selectedCount) {
        selectedCount.textContent = this.selectedModels.length;
    }
};

ChessGame.prototype.makeMove = async function(move) {
    try {
        const response = await fetch('/api/play_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ move: move })
        });
        
        const result = await response.json();
        
        if (result.success) {
            this.currentFen = result.fen;
            this.gameHistory = result.history;
            
            // Update display
            this.renderBoard();
            this.updateGameStatus();
            
            // Get new Stockfish evaluation for the position
            if (result.stockfish_evaluation) {
                this.currentAnalysis.stockfish_eval = result.stockfish_evaluation;
                await this.updateAnalysisDisplay();
            }
            
            // Check for game over
            if (result.game_over) {
                alert(`Game Over: ${result.result}`);
            }
            
        } else {
            alert(`Invalid move: ${result.error || 'Unknown error'}`);
        }
        
    } catch (error) {
        console.error('Error making move:', error);
        alert('Error making move');
    }
};

ChessGame.prototype.playStockfishMove = async function() {
    if (this.isAnalyzing) return;
    
    const stockfishBtn = document.getElementById('stockfishBtn');
    stockfishBtn.disabled = true;
    stockfishBtn.textContent = 'Thinking...';
    
    try {
        const response = await fetch('/api/stockfish_move', { method: 'POST' });
        const data = await response.json();
        
        if (data.move) {
            await this.makeMove(data.move);
        } else {
            alert('Stockfish could not suggest a move');
        }
    } catch (error) {
        console.error('Error getting Stockfish move:', error);
        alert('Error getting Stockfish move');
    } finally {
        stockfishBtn.disabled = false;
        stockfishBtn.textContent = 'Play Stockfish Move';
    }
};

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    if (!window.game) return;
    
    // Space bar to analyze
    if (event.code === 'Space' && !event.target.matches('input, textarea')) {
        event.preventDefault();
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (!analyzeBtn.disabled) {
            window.game.analyzePosition();
        }
    }
    
    // R key to reset
    if (event.code === 'KeyR' && !event.target.matches('input, textarea')) {
        event.preventDefault();
        window.game.resetGame();
    }
    
    // S key for Stockfish move
    if (event.code === 'KeyS' && !event.target.matches('input, textarea')) {
        event.preventDefault();
        const stockfishBtn = document.getElementById('stockfishBtn');
        if (!stockfishBtn.disabled) {
            window.game.playStockfishMove();
        }
    }
});