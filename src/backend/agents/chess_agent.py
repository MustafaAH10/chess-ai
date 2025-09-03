"""Individual chess agent for AI model analysis"""

import logging
import re
import time
from typing import Optional, Tuple
import chess
from ..models.schemas import ModelResponse
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


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