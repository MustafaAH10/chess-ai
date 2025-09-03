"""Pydantic schemas for API models"""

from typing import Dict, List, Optional
from pydantic import BaseModel


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


class SingleModelRequest(BaseModel):
    fen: str
    model_name: str
    previous_reasoning: str = ""


class PlayMoveRequest(BaseModel):
    move: str


class LegalMovesRequest(BaseModel):
    fen: str