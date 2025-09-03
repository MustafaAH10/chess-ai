"""Main chess system coordinating multiple agents"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from fastapi import WebSocket
import chess

from ..models.schemas import ConsensusResult, ModelResponse
from ..engines.stockfish import ChessEngine
from ..agents.ollama_client import OllamaClient
from ..agents.chess_agent import ChessAgent

logger = logging.getLogger(__name__)


class MultiAgentChessSystem:
    """Main system coordinating multiple chess agents"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.chess_engine = ChessEngine()
        self.current_game = chess.Board()
        self.game_history = []
        self.active_connections: Set[WebSocket] = set()
        
    async def initialize(self):
        """Initialize the system"""
        await self.chess_engine.start_engine()
        logger.info("Multi-Agent Chess System initialized")
    
    async def shutdown(self):
        """Shutdown the system"""
        await self.chess_engine.stop_engine()
    
    def reset_game(self):
        """Reset to a new game"""
        self.current_game = chess.Board()
        self.game_history = []
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a WebSocket connection"""
        self.active_connections.add(websocket)
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
    
    async def broadcast_update(self, message: dict):
        """Broadcast update to all connected WebSocket clients"""
        if not self.active_connections:
            return
            
        disconnected = set()
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.active_connections.discard(ws)

    async def get_collaborative_move_streaming(self, fen: str, selected_models: List[str], enable_debate: bool = True) -> ConsensusResult:
        """Get move through sequential model collaboration with optional debate - with real-time updates"""
        board = chess.Board(fen)
        model_responses = []
        reasoning_chain = ""
        
        # Send initial status
        await self.broadcast_update({
            "type": "analysis_start",
            "phase": "initialization",
            "message": "Starting analysis...",
            "selected_models": selected_models
        })
        
        # Get Stockfish evaluation first
        await self.broadcast_update({
            "type": "stockfish_start",
            "message": "Getting Stockfish evaluation..."
        })
        
        stockfish_eval = await self.chess_engine.get_evaluation(board)
        
        await self.broadcast_update({
            "type": "stockfish_complete",
            "stockfish_evaluation": stockfish_eval,
            "message": f"Stockfish suggests: {stockfish_eval.get('best_move_san', 'Unknown')}"
        })
        
        # Sequential reasoning - models analyze independently first
        await self.broadcast_update({
            "type": "phase_start",
            "phase": "independent_analysis",
            "message": "Phase 1: Independent model analysis"
        })
        
        for i, model_name in enumerate(selected_models):
            await self.broadcast_update({
                "type": "model_start",
                "model_name": model_name,
                "model_index": i,
                "phase": "independent",
                "message": f"Analyzing with {model_name}..."
            })
            
            agent = ChessAgent(model_name, self.ollama_client)
            response = await agent.analyze_position(board, "")
            model_responses.append(response)
            
            # Send individual model response immediately
            await self.broadcast_update({
                "type": "model_complete",
                "model_name": model_name,
                "model_index": i,
                "phase": "independent",
                "response": response.model_dump(),
                "message": f"{model_name} suggests: {response.suggested_move or 'No valid move'}"
            })
        
        # Check for conflicts
        unique_moves = set(r.suggested_move for r in model_responses if r.suggested_move)
        has_conflict = len(unique_moves) > 1
        
        debate_rounds = []
        
        if has_conflict and enable_debate and len(selected_models) > 1:
            await self.broadcast_update({
                "type": "conflict_detected",
                "unique_moves": list(unique_moves),
                "message": f"Conflict detected! Models suggest {len(unique_moves)} different moves. Starting debate..."
            })
            
            # Create debate summary for re-prompting
            move_counts = {}
            for response in model_responses:
                if response.suggested_move:
                    move_counts[response.suggested_move] = move_counts.get(response.suggested_move, 0) + 1
            
            debate_summary = "CONFLICTING MOVE SUGGESTIONS DETECTED:\n"
            for move, count in move_counts.items():
                supporting_models = [r.model_name for r in model_responses if r.suggested_move == move]
                debate_summary += f"- {move}: supported by {', '.join(supporting_models)}\n"
            
            await self.broadcast_update({
                "type": "debate_summary",
                "move_counts": move_counts,
                "debate_summary": debate_summary
            })
            
            # Get debate responses - models see others' reasoning
            await self.broadcast_update({
                "type": "phase_start",
                "phase": "debate",
                "message": "Phase 2: Model debate"
            })
            
            debate_responses = []
            for i, model_name in enumerate(selected_models):
                await self.broadcast_update({
                    "type": "model_start",
                    "model_name": model_name,
                    "model_index": i,
                    "phase": "debate",
                    "message": f"Getting debate response from {model_name}..."
                })
                
                # Show other models' reasoning
                other_reasoning = ""
                for j, other_response in enumerate(model_responses):
                    if i != j:  # Don't show model its own reasoning
                        other_reasoning += f"\n--- {other_response.model_name} ---\n"
                        other_reasoning += f"Move: {other_response.suggested_move}\n"
                        other_reasoning += f"Reasoning: {other_response.reasoning}\n"
                
                debate_prompt = f"{debate_summary}\n\nOTHER MODELS' REASONING:{other_reasoning}\n\nGiven these conflicting views, reconsider your analysis and provide your final recommendation."
                
                agent = ChessAgent(model_name, self.ollama_client)
                debate_response = await agent.analyze_position(board, debate_prompt)
                debate_responses.append(debate_response)
                
                # Send debate response immediately
                await self.broadcast_update({
                    "type": "model_complete",
                    "model_name": model_name,
                    "model_index": i,
                    "phase": "debate",
                    "response": debate_response.model_dump(),
                    "message": f"{model_name} debate response: {debate_response.suggested_move or 'No valid move'}"
                })
            
            debate_rounds.append({
                "round": 1,
                "responses": debate_responses,
                "trigger": "conflicting_moves"
            })
            
            # Use debate responses for final consensus
            model_responses = debate_responses
        
        # Build final reasoning chain
        for i, response in enumerate(model_responses):
            reasoning_chain += f"\n\n--- {response.model_name} ---\n{response.reasoning}"
            if response.suggested_move:
                reasoning_chain += f"\nSuggested move: {response.suggested_move} (confidence: {response.confidence})"
        
        # Determine consensus move
        await self.broadcast_update({
            "type": "consensus_start",
            "message": "Calculating consensus..."
        })
        
        final_move, consensus_strength = self._determine_consensus(model_responses, board)
        
        # If debate occurred but produced no valid moves, fall back to independent responses
        if has_conflict and enable_debate and final_move is None:
            logger.warning("Debate responses contained no valid moves, falling back to independent responses")
            # Use the original independent responses before debate started
            independent_responses = []
            for response in model_responses:
                # Look for the independent responses (they should still be in model_responses)
                if hasattr(response, 'phase') and getattr(response, 'phase', None) != 'debate':
                    independent_responses.append(response)
            
            if independent_responses:
                final_move, consensus_strength = self._determine_consensus(independent_responses, board)
                # Update to show we fell back to independent analysis
                await self.broadcast_update({
                    "type": "fallback_consensus",
                    "message": "Using independent analysis due to invalid debate responses"
                })
        
        result = ConsensusResult(
            final_move=final_move,
            consensus_strength=consensus_strength,
            model_responses=model_responses,
            reasoning_chain=reasoning_chain,
            stockfish_evaluation=stockfish_eval,
            debate_rounds=debate_rounds,
            had_conflict=has_conflict
        )
        
        # Send final consensus
        await self.broadcast_update({
            "type": "analysis_complete",
            "final_result": result.model_dump(),
            "message": f"Final consensus: {final_move} ({consensus_strength:.1f}% agreement)"
        })
        
        return result
    
    def _determine_consensus(self, responses: List[ModelResponse], board: chess.Board) -> Tuple[Optional[str], float]:
        """Determine final move from model responses"""
        # Count votes for each move, weighted by confidence
        move_scores = {}
        total_confidence = 0
        
        for response in responses:
            if response.suggested_move:
                move = response.suggested_move
                if move not in move_scores:
                    move_scores[move] = 0
                move_scores[move] += response.confidence
                total_confidence += response.confidence
        
        if not move_scores:
            # No valid moves suggested, return a random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return str(legal_moves[0]), 0.0
            return None, 0.0
        
        # Get move with highest score
        best_move = max(move_scores.items(), key=lambda x: x[1])
        final_move = best_move[0]
        consensus_strength = (best_move[1] / total_confidence) * 100 if total_confidence > 0 else 0
        
        return final_move, consensus_strength
    
    async def play_move(self, move_input: str) -> Dict:
        """Play a move and get updated game state"""
        try:
            # Try to parse as SAN first, then UCI
            move = None
            move_san = None
            
            try:
                # Try SAN format first (e.g., "e4", "Nf3")
                move = self.current_game.parse_san(move_input)
                move_san = move_input
            except:
                try:
                    # Try UCI format (e.g., "e2e4")
                    move = chess.Move.from_uci(move_input)
                    if move in self.current_game.legal_moves:
                        move_san = self.current_game.san(move)
                    else:
                        raise ValueError("Invalid UCI move")
                except:
                    raise ValueError(f"Unable to parse move: {move_input}")
            
            self.current_game.push(move)
            self.game_history.append(move_san)
            
            # Check game state
            game_over = self.current_game.is_game_over()
            result = None
            
            if game_over:
                if self.current_game.is_checkmate():
                    winner = "White" if self.current_game.turn == chess.BLACK else "Black"
                    result = f"{winner} wins by checkmate"
                elif self.current_game.is_stalemate():
                    result = "Draw by stalemate"
                elif self.current_game.is_insufficient_material():
                    result = "Draw by insufficient material"
                else:
                    result = "Draw"
            
            # Get updated Stockfish evaluation after the move
            stockfish_eval = await self.chess_engine.get_evaluation(self.current_game)

            return {
                "success": True,
                "fen": self.current_game.fen(),
                "move_played": move_san,
                "move_uci": str(move),
                "game_over": game_over,
                "result": result,
                "history": self.game_history.copy(),
                "stockfish_evaluation": stockfish_eval
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_stockfish_move(self) -> Optional[str]:
        """Get Stockfish's move for the current position"""
        return await self.chess_engine.get_best_move(self.current_game)
    
    async def get_stockfish_evaluation(self) -> Dict:
        """Get Stockfish evaluation for the current position"""
        return await self.chess_engine.get_evaluation(self.current_game)