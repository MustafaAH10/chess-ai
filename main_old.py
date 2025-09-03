#!/usr/bin/env python3
"""
Multi-Agent Chess System with Sequential Model Reasoning
Uses small models (<3B) loaded one-by-one for collaborative chess analysis
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

import chess
import chess.engine
import chess.pgn
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoveRequest(BaseModel):
    fen: str
    selected_models: List[str]

class GameState(BaseModel):
    fen: str
    moves: List[str]
    game_over: bool
    result: Optional[str]

class ModelResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    reasoning: str
    suggested_move: Optional[str]
    confidence: int
    thinking_time: float

class ConsensusResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    final_move: Optional[str]
    consensus_strength: float
    model_responses: List[ModelResponse]
    reasoning_chain: str
    stockfish_evaluation: Optional[Dict] = None
    debate_rounds: List[Dict] = []
    had_conflict: bool = False

class ChessEngine:
    """Handles Stockfish integration"""
    
    def __init__(self, engine_path: str = "stockfish"):
        self.engine_path = engine_path
        self.engine = None
        
    async def start_engine(self):
        """Start Stockfish engine"""
        try:
            transport, self.engine = await chess.engine.popen_uci(self.engine_path)
            logger.info("Stockfish engine started successfully")
        except Exception as e:
            logger.error(f"Failed to start Stockfish: {e}")
            self.engine = None
    
    async def stop_engine(self):
        """Stop Stockfish engine"""
        if self.engine:
            await self.engine.quit()
            self.engine = None
    
    async def get_best_move(self, board: chess.Board, time_limit: float = 1.0) -> Optional[str]:
        """Get Stockfish's best move"""
        if not self.engine:
            return None
            
        try:
            result = await self.engine.play(board, chess.engine.Limit(time=time_limit))
            return str(result.move) if result.move else None
        except Exception as e:
            logger.error(f"Engine move error: {e}")
            return None
    
    async def get_evaluation(self, board: chess.Board, time_limit: float = 0.5) -> Dict:
        """Get Stockfish evaluation of the position"""
        if not self.engine:
            return {"score": "Engine not available", "best_move": None, "pv": [], "depth": 0}
            
        try:
            # Get analysis with more explicit parameters
            info = await self.engine.analyse(
                board, 
                chess.engine.Limit(time=time_limit, depth=10),
                info=chess.engine.INFO_ALL
            )
            
            score = info.get("score")
            pv = info.get("pv", [])
            best_move = str(pv[0]) if pv else None
            depth = info.get("depth", 0)
            
            # Format score more carefully
            score_text = "Unknown"
            if score:
                # Handle different perspectives (White/Black)
                relative_score = score.white() if board.turn == chess.WHITE else score.black()
                
                if relative_score.is_mate():
                    mate_in = relative_score.mate()
                    if mate_in is not None:
                        score_text = f"Mate in {abs(mate_in)}" if mate_in > 0 else f"Mated in {abs(mate_in)}"
                    else:
                        score_text = "Mate"
                else:
                    centipawns = relative_score.score()
                    if centipawns is not None:
                        score_text = f"{centipawns / 100:+.2f}"  # + for positive, - for negative
            
            # Safe SAN conversion with error handling
            best_move_san = None
            pv_san = []
            
            if pv:
                try:
                    # Create a copy of board for move validation
                    temp_board = board.copy()
                    if pv[0] in temp_board.legal_moves:
                        best_move_san = temp_board.san(pv[0])
                    
                    # Convert PV to SAN safely
                    temp_board = board.copy()
                    for move in pv[:3]:
                        if move in temp_board.legal_moves:
                            pv_san.append(temp_board.san(move))
                            temp_board.push(move)
                        else:
                            break  # Stop if we encounter illegal move
                except Exception as e:
                    logger.error(f"Error converting moves to SAN: {e}")
                    best_move_san = None
                    pv_san = []

            return {
                "score": score_text,
                "best_move": best_move,
                "best_move_san": best_move_san,
                "pv": [str(move) for move in pv[:3]] if pv else [],  # First 3 moves of principal variation
                "pv_san": pv_san,  # SAN version with safe conversion
                "depth": depth
            }
            
        except Exception as e:
            logger.error(f"Engine evaluation error: {e}")
            return {"score": f"Error: {str(e)}", "best_move": None, "pv": [], "depth": 0}

class OllamaClient:
    """Handles Ollama API communication with model loading"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_model = None
    
    async def load_model(self, model_name: str):
        """Load a specific model into memory"""
        if self.current_model == model_name:
            return True
            
        try:
            # Unload current model if any
            if self.current_model:
                logger.info(f"Unloading model: {self.current_model}")
                # Ollama automatically manages memory, but we can be explicit
                
            logger.info(f"Loading model: {model_name}")
            # Make a small request to ensure model is loaded
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"max_tokens": 1}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_model = model_name
                logger.info(f"Successfully loaded model: {model_name}")
                return True
            else:
                logger.error(f"Failed to load model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def generate_response(self, model_name: str, prompt: str) -> str:
        """Generate response from specified model"""
        # Ensure model is loaded
        if not await self.load_model(model_name):
            raise Exception(f"Failed to load model: {model_name}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available small models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()["models"]
                # Filter for small models only
                small_models = []
                for model in models:
                    name = model["name"]
                    # Include models with 1b, 1.5b, 2b, or 3b in name
                    if any(size in name.lower() for size in ["1b", "1.5b", "2b", "3b"]):
                        # Exclude larger models
                        if not any(large in name.lower() for large in ["7b", "8b", "70b", "13b"]):
                            small_models.append(name)
                return small_models
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

class ChessAgent:
    """Individual chess agent for a specific model"""
    
    def __init__(self, model_name: str, ollama_client: OllamaClient):
        self.model_name = model_name
        self.ollama_client = ollama_client
        self.display_name = self._get_display_name()
    
    def _get_display_name(self) -> str:
        """Get friendly display name for model"""
        name_map = {
            'llama3.2:1b': 'Llama 3.2 1B',
            'qwen2.5:1.5b': 'Qwen 2.5 1.5B',
            'gemma2:2b': 'Gemma 2 2B'
        }
        return name_map.get(self.model_name, self.model_name)
    
    def _create_prompt(self, board: chess.Board, previous_reasoning: str = "") -> str:
        """Create reasoning prompt for the model"""
        
        # Get legal moves in SAN (Standard Algebraic Notation)
        legal_moves_san = []
        for move in board.legal_moves:
            legal_moves_san.append(board.san(move))
        
        prompt = f"""You are a chess AI analyzing the current position. Your task is to suggest the best move and explain your reasoning.

CURRENT POSITION:
FEN: {board.fen()}
Turn: {"White" if board.turn == chess.WHITE else "Black"} to move

BOARD:
{board.unicode()}

LEGAL MOVES (Standard Algebraic Notation): {', '.join(legal_moves_san[:20])}{'...' if len(legal_moves_san) > 20 else ''}

GAME STATUS:
- Material: {self._calculate_material(board)}
- Game phase: {self._determine_phase(board)}
- In check: {"Yes" if board.is_check() else "No"}
"""

        if previous_reasoning:
            prompt += f"""
PREVIOUS AI ANALYSIS:
{previous_reasoning}

Consider the previous analysis but provide your own independent assessment. You may agree or disagree.
"""

        prompt += f"""
INSTRUCTIONS:
1. Analyze the position thoroughly
2. Consider tactical opportunities, positional factors, and strategic goals
3. Evaluate the top 3-5 candidate moves
4. You MUST choose one move - failure to provide a valid move is not acceptable
5. Rate your confidence level (1-10)

CRITICAL: You are REQUIRED to select exactly ONE move from the legal moves list below.
LEGAL MOVES (choose ONE): {', '.join(legal_moves_san)}

RESPOND IN THIS EXACT FORMAT:
ANALYSIS: [Your detailed analysis of the position and key factors]
MOVE: <move>EXACT_MOVE_HERE</move>
CONFIDENCE: [Number from 1-10]

MOVE FORMAT REQUIREMENTS:
- You MUST wrap your move in <move> tags: <move>e4</move>
- Your move must be copied EXACTLY from this list: {', '.join(legal_moves_san[:15])}{'...' if len(legal_moves_san) > 15 else ''}

Example valid responses:
- If legal moves include "e4": MOVE: <move>e4</move>
- If legal moves include "Nf3": MOVE: <move>Nf3</move>
- If legal moves include "O-O": MOVE: <move>O-O</move>
- If legal moves include "Qxd7+": MOVE: <move>Qxd7+</move>

CRITICAL: The <move> tags are MANDATORY. Without them, your move will not be detected.
FAILURE TO USE <move> TAGS AND PROVIDE A VALID MOVE IS UNACCEPTABLE.
"""
        return prompt
    
    def _calculate_material(self, board: chess.Board) -> str:
        """Calculate material balance"""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                 chess.ROOK: 5, chess.QUEEN: 9}
        
        white_material = sum(values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE and piece.piece_type != chess.KING)
        black_material = sum(values.get(piece.piece_type, 0) 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK and piece.piece_type != chess.KING)
        
        balance = white_material - black_material
        if balance > 0:
            return f"White +{balance}"
        elif balance < 0:
            return f"Black +{abs(balance)}"
        else:
            return "Equal"
    
    def _determine_phase(self, board: chess.Board) -> str:
        """Determine game phase"""
        piece_count = len([p for p in board.piece_map().values() 
                          if p.piece_type != chess.PAWN])
        
        if piece_count > 20:
            return "Opening"
        elif piece_count > 10:
            return "Middlegame"
        else:
            return "Endgame"
    
    async def analyze_position(self, board: chess.Board, previous_reasoning: str = "") -> ModelResponse:
        """Analyze position and return structured response"""
        start_time = time.time()
        
        try:
            prompt = self._create_prompt(board, previous_reasoning)
            response_text = await self.ollama_client.generate_response(self.model_name, prompt)
            
            # Parse response
            move, confidence, analysis, reasoning = self._parse_response(response_text, board)
            
            thinking_time = time.time() - start_time
            
            return ModelResponse(
                model_name=self.display_name,
                reasoning=f"ANALYSIS: {analysis}\n\nREASONING: {reasoning}",
                suggested_move=move,
                confidence=confidence,
                thinking_time=thinking_time
            )
            
        except Exception as e:
            logger.error(f"Analysis error for {self.model_name}: {e}")
            return ModelResponse(
                model_name=self.display_name,
                reasoning=f"Error during analysis: {str(e)}",
                suggested_move=None,
                confidence=1,
                thinking_time=time.time() - start_time
            )
    
    def _parse_response(self, response: str, board: chess.Board) -> Tuple[Optional[str], int, str, str]:
        """Parse model response to extract components with enhanced move parsing"""
        # Extract sections using regex
        analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=MOVE:|$)', response, re.DOTALL)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        
        analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
        confidence = int(confidence_match.group(1)) if confidence_match else 5
        confidence = max(1, min(10, confidence))  # Clamp to 1-10
        
        # Enhanced move parsing - prioritize <move> tags, fallback to old method
        move = self._extract_move_from_response(response, board)
        
        # Use analysis as the reasoning since we removed the separate REASONING section
        reasoning = analysis
        
        return move, confidence, analysis, reasoning
    
    def _extract_move_from_response(self, response: str, board: chess.Board) -> Optional[str]:
        """Extract and validate move from response with multiple parsing strategies"""
        
        # Strategy 1: Look for <move> tags (preferred)
        move_tag_match = re.search(r'<move>\s*([^<>\s]+)\s*</move>', response, re.IGNORECASE)
        if move_tag_match:
            candidate_move = move_tag_match.group(1).strip()
            validated_move = self._validate_move(candidate_move, board)
            if validated_move:
                logger.info(f"Move extracted from <move> tags: {validated_move}")
                return validated_move
            else:
                logger.warning(f"Move in <move> tags is invalid: {candidate_move}")
        
        # Strategy 2: Look for MOVE: line (fallback)
        move_line_match = re.search(r'MOVE:\s*([^\n\r]+)', response)
        if move_line_match:
            move_line = move_line_match.group(1).strip()
            
            # Try to extract move from the line (might contain <move> tags or just text)
            tag_in_line = re.search(r'<move>\s*([^<>\s]+)\s*</move>', move_line, re.IGNORECASE)
            if tag_in_line:
                candidate_move = tag_in_line.group(1).strip()
            else:
                candidate_move = move_line
            
            validated_move = self._validate_move(candidate_move, board)
            if validated_move:
                logger.info(f"Move extracted from MOVE line: {validated_move}")
                return validated_move
            else:
                logger.warning(f"Move in MOVE line is invalid: {candidate_move}")
        
        # Strategy 3: Look for any chess move pattern in the response (desperate fallback)
        legal_moves_san = [board.san(move) for move in board.legal_moves]
        
        # Sort by length (descending) to match longer moves first (e.g., "Qxd7+" before "Qd7")
        legal_moves_san.sort(key=len, reverse=True)
        
        for legal_move in legal_moves_san:
            # Escape special regex characters in the move
            escaped_move = re.escape(legal_move)
            # Look for the move as a whole word
            pattern = r'\b' + escaped_move + r'\b'
            if re.search(pattern, response):
                logger.info(f"Move found by pattern matching: {legal_move}")
                return legal_move
        
        logger.error("No valid move found in response using any strategy")
        return None
    
    def _validate_move(self, move_str: str, board: chess.Board) -> Optional[str]:
        """Validate if move string is legal"""
        try:
            # Try parsing as SAN (Standard Algebraic Notation)
            move = board.parse_san(move_str)
            return move_str
        except:
            # Try common variations
            variations = [
                move_str.strip(),
                move_str.replace('0', 'O'),  # Convert 0-0 to O-O
                move_str.upper(),
                move_str.lower()
            ]
            
            for variant in variations:
                try:
                    move = board.parse_san(variant)
                    return variant
                except:
                    continue
        
        return None

class MultiAgentChessSystem:
    """Main system coordinating multiple chess agents"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.chess_engine = ChessEngine()
        self.current_game = chess.Board()
        self.game_history = []
        self.active_connections = set()
        
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

# Global system instance
chess_system = MultiAgentChessSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await chess_system.initialize()
    yield
    await chess_system.shutdown()

app = FastAPI(title="Multi-Agent Chess System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading page: {e}</h1>", status_code=500)

@app.get("/api/models")
async def get_available_models():
    """Get list of available small models"""
    models = chess_system.ollama_client.get_available_models()
    return {"models": models}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()
    await chess_system.add_websocket_connection(websocket)
    
    try:
        while True:
            # Keep connection alive - we can add ping/pong here if needed
            data = await websocket.receive_text()
            # Echo back for now - can be used for commands later
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
    finally:
        await chess_system.remove_websocket_connection(websocket)

class SingleModelRequest(BaseModel):
    fen: str
    model_name: str
    previous_reasoning: str = ""

@app.post("/api/analyze_model")
async def analyze_single_model(request: SingleModelRequest):
    """Analyze position with a single model"""
    try:
        board = chess.Board(request.fen)
        agent = ChessAgent(request.model_name, chess_system.ollama_client)
        response = await agent.analyze_position(board, request.previous_reasoning)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model analysis failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_position(request: MoveRequest):
    """Get collaborative analysis from selected models with real-time updates"""
    try:
        result = await chess_system.get_collaborative_move_streaming(request.fen, request.selected_models)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze_sync")
async def analyze_position_sync(request: MoveRequest):
    """Get collaborative analysis from selected models (legacy non-streaming)"""
    try:
        result = await chess_system.get_collaborative_move(request.fen, request.selected_models)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

class PlayMoveRequest(BaseModel):
    move: str

@app.post("/api/play_move")
async def play_move(request: PlayMoveRequest):
    """Play a move and update game state"""
    result = await chess_system.play_move(request.move)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/stockfish_move")
async def get_stockfish_move():
    """Get Stockfish's best move"""
    move = await chess_system.get_stockfish_move()
    return {"move": move}

@app.get("/api/stockfish_eval")
async def get_stockfish_evaluation():
    """Get Stockfish evaluation"""
    evaluation = await chess_system.get_stockfish_evaluation()
    return evaluation

@app.post("/api/reset_game")
async def reset_game():
    """Reset to a new game"""
    chess_system.reset_game()
    return {
        "success": True,
        "fen": chess_system.current_game.fen(),
        "history": []
    }

@app.get("/api/game_state")
async def get_game_state():
    """Get current game state"""
    return {
        "fen": chess_system.current_game.fen(),
        "history": chess_system.game_history.copy(),
        "game_over": chess_system.current_game.is_game_over()
    }

class LegalMovesRequest(BaseModel):
    fen: str

@app.post("/api/legal_moves")
async def get_legal_moves(request: LegalMovesRequest):
    """Get legal moves for a position in SAN notation"""
    try:
        board = chess.Board(request.fen)
        legal_moves_san = []
        for move in board.legal_moves:
            legal_moves_san.append(board.san(move))
        return {"moves": legal_moves_san}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)