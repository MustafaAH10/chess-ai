/**
 * UI management for chess analysis display
 */
ChessGame.prototype.updateAnalysisDisplay = async function() {
    const content = document.getElementById('analysisContent');
    if (!content) return;

    // Update Stockfish evaluation in its dedicated section below the board
    this.updateStockfishDisplay();

    let html = '';

    // Always show the common prompt first
    html += await this.generatePromptDisplay();

    // Always show analysis history if available
    if (this.analysisHistory.length > 0) {
        html += '<div class="analysis-history-section">';
        html += '<h3>📚 Analysis History</h3>';
        
        for (const analysis of this.analysisHistory) {
            html += `
                <div class="history-item">
                    <div class="history-header">
                        <strong>Move #${analysis.moveNumber}</strong>
                        <span class="timestamp">${analysis.timestamp}</span>
                    </div>
                    <div class="selected-move">
                        <strong>Selected:</strong> ${analysis.selectedMove} 
                        <span class="consensus">(${analysis.consensusStrength.toFixed(1)}% consensus)</span>
                        ${analysis.hadDebate ? ' <span class="debate-indicator">🗣️ Debate</span>' : ''}
                    </div>
                </div>
            `;
        }
        
        html += '</div><hr>';
    }

    // Current position section
    html += '<div class="current-analysis-section">';
    html += '<h3>🔬 Current Position Analysis</h3>';

    // Stockfish evaluation (always show if available)
    if (this.currentAnalysis.stockfish_eval) {
        const eval_data = this.currentAnalysis.stockfish_eval;
        html += `
            <div class="stockfish-eval">
                <h4>🏁 Stockfish Evaluation</h4>
                <div class="eval-info">
                    <strong>Score:</strong> ${eval_data.score} 
                    (Depth ${eval_data.depth})
                </div>
                <div class="best-move">
                    <strong>Best Move:</strong> ${eval_data.best_move_san || eval_data.best_move || 'Unknown'}
                </div>
                ${eval_data.pv_san && eval_data.pv_san.length > 0 ? 
                    `<div class="principal-variation"><strong>Line:</strong> ${eval_data.pv_san.join(' ')}</div>` : 
                    ''
                }
            </div>
        `;
    }

    // Model analysis section based on current phase
    switch (this.currentAnalysis.phase) {
        case 'idle':
            html += '<div class="phase-info">Ready for analysis. Select models and click "Analyze Position".</div>';
            break;

        case 'starting':
            html += '<div class="loading"><div class="spinner"></div>Initializing analysis...</div>';
            break;

        case 'stockfish_loading':
            html += '<div class="loading"><div class="spinner"></div>Getting Stockfish evaluation...</div>';
            break;

        case 'independent_analysis':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            break;

        case 'analyzing_llama3.2:1b':
        case 'analyzing_qwen2.5:1.5b':
        case 'analyzing_gemma2:2b':
            const currentModel = this.currentAnalysis.phase.replace('analyzing_', '');
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            html += `<div class="loading"><div class="spinner"></div>Analyzing with ${currentModel}...</div>`;
            break;

        case 'debate_reasoning':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            
            if (this.currentAnalysis.debate_reasoning) {
                html += '<div class="debate-reasoning-section">';
                html += '<h4>🎯 Debate Decision</h4>';
                html += `<div class="reasoning">${this.currentAnalysis.debate_reasoning.reason}</div>`;
                if (this.currentAnalysis.debate_reasoning.breakdown) {
                    html += `<div class="breakdown"><pre>${this.currentAnalysis.debate_reasoning.breakdown}</pre></div>`;
                }
                html += '</div>';
            }
            break;

        case 'conflict_detected':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            html += '<div class="conflict-detected">⚡ Conflict detected! Starting debate phase...</div>';
            break;

        case 'debate':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            html += '<div class="phase-info">🗣️ Phase 2: Model Debate</div>';
            html += this.renderModelResponses('debate');
            break;

        case 'debating_llama3.2:1b':
        case 'debating_qwen2.5:1.5b':
        case 'debating_gemma2:2b':
            const debatingModel = this.currentAnalysis.phase.replace('debating_', '');
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            html += '<div class="phase-info">🗣️ Phase 2: Model Debate</div>';
            html += this.renderModelResponses('debate');
            html += `<div class="loading"><div class="spinner"></div>Getting debate response from ${debatingModel}...</div>`;
            break;

        case 'calculating_consensus':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            
            const debateResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'debate');
            if (debateResponses.length > 0) {
                html += '<div class="phase-info">🗣️ Phase 2: Model Debate</div>';
                html += this.renderModelResponses('debate');
            }
            
            html += '<div class="loading"><div class="spinner"></div>Calculating final consensus...</div>';
            break;

        case 'complete':
            html += '<div class="phase-info">🤖 Phase 1: Independent Model Analysis</div>';
            html += this.renderModelResponses('independent');
            
            const finalDebateResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'debate');
            if (finalDebateResponses.length > 0) {
                html += '<div class="phase-info">🗣️ Phase 2: Model Debate</div>';
                html += this.renderModelResponses('debate');
            }
            
            if (this.currentAnalysis.final_result) {
                html += this.renderFinalResult();
            }
            break;

        default:
            html += `<div class="phase-info">Status: ${this.currentAnalysis.phase}</div>`;
            html += this.renderModelResponses('independent');
            html += this.renderModelResponses('debate');
            break;
    }

    html += '</div>'; // Close current-analysis-section
    content.innerHTML = html;
};

ChessGame.prototype.renderModelResponses = function(phase) {
    const responses = this.currentAnalysis.model_responses.filter(r => r.phase === phase);
    if (responses.length === 0) return '';

    let html = '<div class="model-responses">';
    
    for (const response of responses) {
        const displayName = this.getModelDisplayName(response.model_name);
        const statusClass = response.suggested_move ? 'completed' : 'error';
        
        html += `
            <div class="model-box ${statusClass}">
                <div class="model-header">
                    <h4>${displayName}</h4>
                    <div class="model-status">
                        ${response.thinking_time ? `⏱️ ${response.thinking_time.toFixed(1)}s` : ''}
                    </div>
                </div>
                
                <div class="model-response">
                    <div class="suggested-move">
                        <strong>Move:</strong> 
                        <span class="move">${response.suggested_move || 'No valid move'}</span>
                        ${response.confidence ? `<span class="confidence">(${response.confidence}/10)</span>` : ''}
                    </div>
                    
                    <div class="reasoning">
                        <strong>Analysis:</strong>
                        <div class="reasoning-text">${response.reasoning || 'No analysis provided'}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
};

ChessGame.prototype.renderFinalResult = function() {
    const result = this.currentAnalysis.final_result;
    if (!result) return '';

    let html = '<div class="final-result">';
    html += '<h4>🎯 Final Consensus</h4>';
    
    html += `
        <div class="consensus-info">
            <div class="final-move">
                <strong>Selected Move:</strong> 
                <span class="move">${result.final_move || 'None'}</span>
            </div>
            <div class="consensus-strength">
                <strong>Consensus Strength:</strong> ${result.consensus_strength.toFixed(1)}%
            </div>
            <div class="analysis-stats">
                ${result.had_conflict ? '⚡ Had conflict' : '✅ No conflict'} | 
                ${result.had_debate ? '🗣️ Debate occurred' : '🤝 No debate needed'}
            </div>
        </div>
    `;

    if (result.debate_decision && result.debate_decision.reason) {
        html += `
            <div class="debate-summary">
                <strong>Decision Logic:</strong> ${result.debate_decision.reason}
            </div>
        `;
    }

    // ADD THE CRITICAL PLAY MOVE BUTTON!
    if (result.final_move) {
        html += `
            <div class="play-move-section">
                <button class="btn btn-primary play-move-btn" onclick="game.makeMove('${result.final_move}')" 
                        style="background: rgba(255,255,255,0.9); color: #2c3e50; border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; font-size: 1.1em; margin-top: 15px;">
                    ▶️ Play Move: ${result.final_move}
                </button>
            </div>
        `;
    }

    html += '</div>';
    return html;
};

ChessGame.prototype.getModelDisplayName = function(modelName) {
    const nameMap = {
        'llama3.2:1b': 'Llama 3.2 1B',
        'qwen2.5:1.5b': 'Qwen 2.5 1.5B', 
        'gemma2:2b': 'Gemma 2 2B',
        'Llama 3.2 1B': 'Llama 3.2 1B',
        'Qwen 2.5 1.5B': 'Qwen 2.5 1.5B',
        'Gemma 2 2B': 'Gemma 2 2B'
    };
    return nameMap[modelName] || modelName;
};

ChessGame.prototype.formatReasoning = function(reasoning) {
    if (!reasoning) return 'No reasoning provided';
    
    // Convert newlines to HTML breaks and preserve formatting
    // Remove deduplication - just show analysis (reasoning contains the same info)
    return reasoning
        .replace(/\n/g, '<br>')
        .replace(/ANALYSIS:\s*/g, '<strong>Analysis:</strong> ')
        .replace(/REASONING:\s*.*?(?=<br>|$)/g, '') // Remove duplicate reasoning section
        .replace(/MOVE:\s*/g, '<br><strong>Move:</strong> ')
        .replace(/CONFIDENCE:\s*/g, '<br><strong>Confidence:</strong> ');
};

ChessGame.prototype.updateStockfishDisplay = function() {
    const stockfishDiv = document.getElementById('stockfishEval');
    if (!stockfishDiv) return;

    if (this.currentAnalysis.stockfish_eval) {
        const eval_data = this.currentAnalysis.stockfish_eval;
        stockfishDiv.innerHTML = `
            <h4>🏁 Stockfish Evaluation</h4>
            <div><strong>Score:</strong> ${eval_data.score} (Depth ${eval_data.depth})</div>
            <div><strong>Best Move:</strong> ${eval_data.best_move_san || eval_data.best_move || 'Unknown'}</div>
            ${eval_data.pv_san && eval_data.pv_san.length > 0 ? 
                `<div><strong>Line:</strong> ${eval_data.pv_san.join(' ')}</div>` : 
                ''
            }
        `;
        stockfishDiv.style.display = 'block';
    } else {
        stockfishDiv.innerHTML = '<div>🏁 Stockfish: Evaluating...</div>';
        stockfishDiv.style.display = 'block';
    }
};

ChessGame.prototype.generatePromptDisplay = async function() {
    const isWhiteToMove = this.currentFen.includes(' w ');
    
    // Get legal moves from backend
    let legalMoves = [];
    try {
        const response = await fetch('/api/legal_moves', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fen: this.currentFen })
        });
        if (response.ok) {
            const data = await response.json();
            legalMoves = data.moves || [];
        }
    } catch (error) {
        console.error('Error fetching legal moves:', error);
    }

    const movesString = legalMoves.length > 0 ? legalMoves.slice(0, 10).join(', ') + (legalMoves.length > 10 ? '...' : '') : 'Error loading moves';
    
    let html = `
        <div class="prompt-display-section">
            <h4>📝 Current Prompt Being Sent to All Models</h4>
            <div class="prompt-text">
                <strong>Position:</strong> ${this.currentFen}<br>
                <strong>Turn:</strong> ${isWhiteToMove ? 'White' : 'Black'} to move<br>
                <strong>Legal Moves:</strong> ${movesString}<br><br>
                <em>Each model receives detailed position analysis instructions and must respond with ANALYSIS, MOVE, and CONFIDENCE.</em>
            </div>
        </div>
    `;

    // If we're in debate phase, show the debate prompt too
    if (this.currentAnalysis.phase && (this.currentAnalysis.phase.includes('debate') || this.currentAnalysis.phase === 'conflict_detected')) {
        html += await this.generateDebatePromptDisplay();
    }

    return html;
};

ChessGame.prototype.generateDebatePromptDisplay = async function() {
    const independentResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'independent');
    if (independentResponses.length === 0) return '';

    let debatePrompt = `
        <div class="debate-prompt-section">
            <h4>🗣️ Additional Debate Prompt Being Sent</h4>
            <div class="prompt-text">
                <strong>CONFLICTING MOVE SUGGESTIONS DETECTED:</strong><br>
    `;

    // Group moves by what models suggested them
    const moveGroups = {};
    for (const response of independentResponses) {
        if (response.suggested_move) {
            if (!moveGroups[response.suggested_move]) {
                moveGroups[response.suggested_move] = [];
            }
            moveGroups[response.suggested_move].push(response.model_name);
        }
    }

    for (const [move, models] of Object.entries(moveGroups)) {
        debatePrompt += `- ${move}: supported by ${models.join(', ')}<br>`;
    }

    debatePrompt += `<br><strong>OTHER MODELS' REASONING:</strong><br>`;
    for (const response of independentResponses) {
        debatePrompt += `<strong>${response.model_name}:</strong> Move ${response.suggested_move}, ${response.reasoning.substring(0, 100)}...<br>`;
    }

    debatePrompt += `<br><em>Given these conflicting views, reconsider your analysis and provide your final recommendation.</em>
            </div>
        </div>
    `;

    return debatePrompt;
};