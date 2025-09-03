/**
 * Analysis management functionality including move consensus and debate logic
 */

ChessGame.prototype.calculateMoveVotes = function(responses) {
    const moveVotes = {};
    let totalModels = 0;
    
    for (const response of responses) {
        if (response.suggested_move) {
            if (!moveVotes[response.suggested_move]) {
                moveVotes[response.suggested_move] = {
                    count: 0,
                    models: [],
                    totalConfidence: 0
                };
            }
            moveVotes[response.suggested_move].count++;
            moveVotes[response.suggested_move].models.push(response.model_name);
            moveVotes[response.suggested_move].totalConfidence += response.confidence;
            totalModels++;
        }
    }
    
    // Calculate percentages
    for (const move in moveVotes) {
        moveVotes[move].percentage = (moveVotes[move].count / totalModels) * 100;
    }
    
    return { moves: moveVotes, totalModels };
};

ChessGame.prototype.shouldEnterDebate = function(moveVotes, totalModels) {
    const moves = Object.entries(moveVotes.moves);
    
    if (moves.length <= 1) {
        return { 
            needsDebate: false, 
            reason: "All models agree on the same move",
            winner: moves[0]?.[0] || null
        };
    }
    
    // Sort by vote count (descending)
    moves.sort((a, b) => b[1].count - a[1].count);
    
    const topMove = moves[0];
    const secondMove = moves[1] || null;
    
    const topPercentage = topMove[1].percentage;
    
    // Clear majority (60%+ for 3+ models, 75%+ for 2 models)
    const majorityThreshold = totalModels === 2 ? 75 : 60;
    
    if (topPercentage >= majorityThreshold) {
        return {
            needsDebate: false,
            reason: `Clear majority: ${topMove[1].count}/${totalModels} models (${topPercentage.toFixed(1)}%) choose ${topMove[0]}`,
            winner: topMove[0],
            breakdown: `${topMove[0]}: ${topMove[1].models.join(', ')}`
        };
    }
    
    // Check for tie (only triggers debate if it's a close tie)
    if (secondMove && topMove[1].count === secondMove[1].count) {
        return {
            needsDebate: true,
            reason: `Tie detected: ${topMove[1].count} models each for ${topMove[0]} and ${secondMove[0]}`,
            breakdown: `${topMove[0]}: ${topMove[1].models.join(', ')}\n${secondMove[0]}: ${secondMove[1].models.join(', ')}`
        };
    }
    
    // Close vote but no clear majority
    return {
        needsDebate: true,
        reason: `No clear majority: ${topMove[1].count}/${totalModels} models (${topPercentage.toFixed(1)}%) choose ${topMove[0]}`,
        breakdown: moves.map(([move, data]) => `${move}: ${data.models.join(', ')}`).join('\n')
    };
};

ChessGame.prototype.calculateConsensus = function(responses) {
    const moveScores = {};
    let totalConfidence = 0;

    for (const response of responses) {
        if (response.suggested_move) {
            moveScores[response.suggested_move] = (moveScores[response.suggested_move] || 0) + response.confidence;
            totalConfidence += response.confidence;
        }
    }

    if (Object.keys(moveScores).length === 0) {
        return { final_move: null, consensus_strength: 0 };
    }

    const bestMove = Object.entries(moveScores).reduce((a, b) => a[1] > b[1] ? a : b);
    const consensusStrength = totalConfidence > 0 ? (bestMove[1] / totalConfidence) * 100 : 0;

    return {
        final_move: bestMove[0],
        consensus_strength: consensusStrength
    };
};

ChessGame.prototype.analyzePosition = async function() {
    if (this.selectedModels.length === 0) {
        alert('Please select at least one model for analysis');
        return;
    }

    if (this.isAnalyzing) return;
    this.isAnalyzing = true;

    const analyzeBtn = document.getElementById('analyzeBtn');
    const stockfishBtn = document.getElementById('stockfishBtn');
    
    analyzeBtn.disabled = true;
    stockfishBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Reset analysis state
    this.currentAnalysis = {
        stockfish_eval: null,
        model_responses: [],
        debate_rounds: [],
        phase: 'starting'
    };

    try {
        // Step 1: Show initial setup
        this.currentAnalysis.phase = 'starting';
        await this.updateAnalysisDisplay();

        // Step 2: Get Stockfish evaluation
        this.currentAnalysis.phase = 'stockfish_loading';
        await this.updateAnalysisDisplay();
        
        const stockfishResponse = await fetch('/api/stockfish_eval');
        if (stockfishResponse.ok) {
            this.currentAnalysis.stockfish_eval = await stockfishResponse.json();
        }

        // Step 3: Independent model analysis - call each model individually
        this.currentAnalysis.phase = 'independent_analysis';
        await this.updateAnalysisDisplay();

        const independentResponses = [];
        for (let i = 0; i < this.selectedModels.length; i++) {
            const modelName = this.selectedModels[i];
            
            // Update UI to show which model is currently working
            this.currentAnalysis.phase = `analyzing_${modelName}`;
            await this.updateAnalysisDisplay();

            try {
                const response = await fetch('/api/analyze_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        fen: this.currentFen,
                        model_name: modelName,
                        previous_reasoning: ""
                    })
                });

                if (response.ok) {
                    const modelResponse = await response.json();
                    modelResponse.phase = 'independent';
                    modelResponse.model_index = i;
                    independentResponses.push(modelResponse);
                    this.currentAnalysis.model_responses.push(modelResponse);
                    
                    // Update UI immediately after each model responds
                    await this.updateAnalysisDisplay();
                }
            } catch (error) {
                console.error(`Error analyzing with ${modelName}:`, error);
            }
        }

        // Step 4: Intelligent debate decision based on vote distribution
        const moveVotes = this.calculateMoveVotes(independentResponses);
        const debateDecision = this.shouldEnterDebate(moveVotes, this.selectedModels.length);

        if (debateDecision.needsDebate) {
            // Show debate reasoning to user
            this.currentAnalysis.phase = 'debate_reasoning';
            this.currentAnalysis.debate_reasoning = debateDecision;
            await this.updateAnalysisDisplay();
            
            // Wait a moment for user to read the reasoning
            await new Promise(resolve => setTimeout(resolve, 2000));
            this.currentAnalysis.phase = 'conflict_detected';
            await this.updateAnalysisDisplay();

            // Prepare debate prompt
            let debatePrompt = "CONFLICTING MOVE SUGGESTIONS DETECTED:\n";
            for (const move in moveVotes.moves) {
                const supportingModels = moveVotes.moves[move].models;
                debatePrompt += `- ${move}: supported by ${supportingModels.join(', ')}\n`;
            }

            // Add other models' reasoning
            debatePrompt += "\nOTHER MODELS' REASONING:\n";
            for (const response of independentResponses) {
                debatePrompt += `--- ${response.model_name} ---\n`;
                debatePrompt += `Move: ${response.suggested_move}\nReasoning: ${response.reasoning}\n\n`;
            }
            
            debatePrompt += "Given these conflicting views, reconsider your analysis and provide your final recommendation.";

            // Run debate phase - call each model individually again
            this.currentAnalysis.phase = 'debate';
            await this.updateAnalysisDisplay();

            for (let i = 0; i < this.selectedModels.length; i++) {
                const modelName = this.selectedModels[i];
                
                // Show which model is debating
                this.currentAnalysis.phase = `debating_${modelName}`;
                await this.updateAnalysisDisplay();

                try {
                    const response = await fetch('/api/analyze_model', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            fen: this.currentFen,
                            model_name: modelName,
                            previous_reasoning: debatePrompt
                        })
                    });

                    if (response.ok) {
                        const debateResponse = await response.json();
                        debateResponse.phase = 'debate';
                        debateResponse.model_index = i;
                        this.currentAnalysis.model_responses.push(debateResponse);
                        
                        // Update UI immediately after each debate response
                        await this.updateAnalysisDisplay();
                    }
                } catch (error) {
                    console.error(`Error getting debate response from ${modelName}:`, error);
                }
            }
        }

        // Step 5: Calculate and show final consensus
        this.currentAnalysis.phase = 'calculating_consensus';
        await this.updateAnalysisDisplay();

        let finalResponses, hadDebate;
        
        if (debateDecision.needsDebate) {
            // Use debate responses
            finalResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'debate');
            hadDebate = true;
        } else {
            // Use independent responses, winner already decided
            finalResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'independent');
            hadDebate = false;
        }

        let consensus;
        if (debateDecision.needsDebate) {
            consensus = this.calculateConsensus(finalResponses);
            
            // If debate responses are all invalid, fall back to independent responses
            if (!consensus.final_move) {
                console.warn('Debate responses contained no valid moves, falling back to independent responses');
                const independentResponses = this.currentAnalysis.model_responses.filter(r => r.phase === 'independent');
                consensus = this.calculateConsensus(independentResponses);
                hadDebate = false; // Mark that debate didn't effectively contribute
            }
        } else {
            consensus = { final_move: debateDecision.winner, consensus_strength: 100 };
        }
        
        this.currentAnalysis.final_result = {
            final_move: consensus.final_move,
            consensus_strength: consensus.consensus_strength,
            had_conflict: moveVotes.moves && Object.keys(moveVotes.moves).length > 1,
            had_debate: hadDebate,
            debate_decision: debateDecision
        };

        this.currentAnalysis.phase = 'complete';
        await this.updateAnalysisDisplay();
        
        // Save this analysis to history before it gets overwritten
        this.saveAnalysisToHistory();

    } catch (error) {
        console.error('Error analyzing position:', error);
        document.getElementById('analysisContent').innerHTML = `<div class="loading">Analysis failed: ${error.message}</div>`;
    } finally {
        this.isAnalyzing = false;
        analyzeBtn.disabled = false;
        stockfishBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Position';
    }
};