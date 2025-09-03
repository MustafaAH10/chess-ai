"""Stockfish chess engine integration"""

import logging
from typing import Dict, Optional
import chess
import chess.engine

logger = logging.getLogger(__name__)


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