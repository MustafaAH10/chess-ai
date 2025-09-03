#!/usr/bin/env python3
"""
Multi-Agent Chess System with Sequential Model Reasoning
Uses small models (<3B) loaded one-by-one for collaborative chess analysis
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import from modular structure
from src.backend.models.schemas import MoveRequest, ModelResponse, ConsensusResult
from src.backend.utils.chess_system import MultiAgentChessSystem
from src.backend.agents.chess_agent import ChessAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Mount static files for the frontend modules
app.mount("/src", StaticFiles(directory="src"), name="static")

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