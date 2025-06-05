from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import chess
import chess.pgn
import requests
import json
import threading
import time
from datetime import datetime
import re
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chess-game-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Available Models
MODELS = [      
    "deepseek-r1:7b",
    "gemma3:4b",
    "llama3.2:3b",
    "mistral:7b",
    "llava"
]

OLLAMA_URL = "http://localhost:11434"

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.current_player = "white"
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move_time = None
        self.move_timeout = 300  # 5 minutes per move
        self.max_moves = 200  # Maximum moves before draw
        self.consecutive_illegal_moves = 0
        self.max_illegal_moves = 3
        
    def validate_move(self, move_uci):
        """Validate a move."""
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                return False, "Illegal move"
                
            current_time = time.time()
            if self.last_move_time and (current_time - self.last_move_time) > self.move_timeout:
                return False, "Move timeout"
                
            return True, "Move valid"
            
        except Exception as e:
            return False, f"Error validating move: {str(e)}"
        
    def make_move(self, move_uci):
        """Make a move with validation."""
        if self.game_over:
            return False, "Game is already over"
            
        is_valid, reason = self.validate_move(move_uci)
        if not is_valid:
            self.consecutive_illegal_moves += 1
            if self.consecutive_illegal_moves >= self.max_illegal_moves:
                self.game_over = True
                self.winner = "black" if self.current_player == "white" else "white"
                return False, f"Too many illegal moves ({self.consecutive_illegal_moves})"
            return False, reason
            
        try:
            move = chess.Move.from_uci(move_uci)
            self.board.push(move)
            self.move_history.append(move_uci)
            self.last_move_time = time.time()
            self.consecutive_illegal_moves = 0
            
            self.current_player = "black" if self.current_player == "white" else "white"
            
            if self.board.is_checkmate():
                self.game_over = True
                self.winner = "black" if self.current_player == "white" else "white"
                return True, "Checkmate"
            elif self.board.is_stalemate():
                self.game_over = True
                self.winner = "draw"
                return True, "Stalemate"
            elif self.board.is_insufficient_material():
                self.game_over = True
                self.winner = "draw"
                return True, "Insufficient material"
            elif len(self.move_history) >= self.max_moves:
                self.game_over = True
                self.winner = "draw"
                return True, "Maximum moves reached"
                
            return True, "Move successful"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def get_game_state(self):
        """Get current game state."""
        return {
            "fen": self.board.fen(),
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "move_history": self.move_history,
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material(),
            "move_count": len(self.move_history)
        }

class OllamaManager:
    def __init__(self):
        self.current_model = None
        self.model_lock = threading.Lock()
        self.load_timeout = 30  # 30 seconds timeout for model loading
        self.multimodal_timeout = 600  # 10 minutes timeout for multimodal models
        self.standard_timeout = 120  # 2 minutes timeout for standard models
        self.save_images = True  # Flag to control image saving
        self.image_dir = "board_images"  # Directory to save images
        
        # Create image directory if it doesn't exist
        if self.save_images:
            os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize game log file
        self.game_log = f"game_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(self.game_log, 'w') as f:
            f.write("=== Chess Game Log ===\n\n")
        
        # Model configurations
        self.model_configs = {
            "deepseek-r1:7b": {
                "temperature": 0.7,
                "top_p": 0.9,
                "supports_cot": True,
                "supports_tools": True,
                "max_tokens": 512,
                "context_window": 4096,
                "supports_multimodal": False
            },
            "gemma3:4b": {
                "temperature": 0.5,
                "top_p": 0.95,
                "supports_cot": False,
                "supports_tools": False,
                "max_tokens": 1024,
                "context_window": 2048,
                "supports_multimodal": True
            },
            "llama3.2:3b": {
                "temperature": 0.7,
                "top_p": 0.9,
                "supports_cot": False,
                "supports_tools": True,
                "max_tokens": 1024,
                "context_window": 2048,
                "supports_multimodal": False
            },
            "mistral:7b": {
                "temperature": 0.7,
                "top_p": 0.9,
                "supports_cot": True,
                "supports_tools": True,
                "max_tokens": 2048,
                "context_window": 4096,
                "supports_multimodal": False
            },
            "llava": {
                "temperature": 0.7,
                "top_p": 0.9,
                "supports_cot": True,
                "supports_tools": False,
                "max_tokens": 2048,
                "context_window": 4096,
                "supports_multimodal": True
            }
        }
    
    def get_model_config(self, model_name):
        return self.model_configs.get(model_name, {
            "temperature": 0.7,
            "top_p": 0.9,
            "supports_cot": False,
            "supports_tools": False,
            "max_tokens": 1024
        })
    
    def ensure_model_loaded(self, model_name):
        """Ensure model is loaded with improved error handling."""
        with self.model_lock:
            if self.current_model == model_name:
                return True
                
            try:
                print(f"Loading model: {model_name}")
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Hello",
                        "stream": False
                    },
                    timeout=self.load_timeout
                )
                
                if response.status_code == 200:
                    self.current_model = model_name
                    print(f"Model {model_name} loaded successfully")
                    return True
                else:
                    print(f"Failed to load model {model_name}: HTTP {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                return False
                
    def unload_current_model(self):
        """Unload the current model."""
        with self.model_lock:
            if self.current_model:
                try:
                    # Send a request to unload the model
                    response = requests.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={
                            "model": self.current_model,
                            "prompt": "",
                            "stream": False
                        },
                        timeout=5
                    )
                    self.current_model = None
                    return True
                except Exception as e:
                    print(f"Error unloading model: {str(e)}")
                    return False
            return True
    
    def generate_board_image(self, board):
        """Generate a base64 encoded image of the chess board with piece letters and square notation."""
        import base64
        from PIL import Image, ImageDraw, ImageFont
        import io
        import time
        
        # Create a 400x400 image with a light background
        img = Image.new('RGB', (400, 400), '#f0d9b5')
        draw = ImageDraw.Draw(img)
        
        # Try to load a default font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            font = None
            small_font = None
        
        square_size = 50
        files = 'abcdefgh'
        ranks = '87654321'
        for rank_idx, rank in enumerate(ranks):
            for file_idx, file in enumerate(files):
                x1 = file_idx * square_size
                y1 = rank_idx * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                # Alternate colors
                if (rank_idx + file_idx) % 2 == 1:
                    draw.rectangle([x1, y1, x2, y2], fill='#b58863')
                # Draw algebraic notation in bottom left
                notation = f"{file}{rank}"
                draw.text((x1 + 2, y2 - 14), notation, fill="#888", font=small_font)
        
        # Draw pieces as letters
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                file_idx = chess.square_file(square)
                rank_idx = 7 - chess.square_rank(square)  # Flip rank for display
                x = file_idx * square_size + square_size // 2
                y = rank_idx * square_size + square_size // 2 - 8
                letter = piece.symbol()
                if piece.color:
                    # White piece: uppercase, white color
                    draw.text((x-10, y), letter.upper(), fill='white', font=font)
                else:
                    # Black piece: lowercase, black color
                    draw.text((x-10, y), letter.lower(), fill='black', font=font)
        
        # Save image if enabled
        if self.save_images:
            timestamp = int(time.time())
            filename = f"{self.image_dir}/board_{timestamp}.png"
            img.save(filename)
            print(f"Saved board image to {filename}")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def log_move(self, move_number, color, model_name, prompt, response, board_image=None):
        """Log move details to the game log file."""
        with open(self.game_log, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Move {move_number} - {color.upper()} ({model_name})\n")
            f.write(f"{'='*50}\n\n")
            
            f.write("PROMPT:\n")
            f.write("-"*20 + "\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("RESPONSE:\n")
            f.write("-"*20 + "\n")
            f.write(response)
            f.write("\n\n")
            
            if board_image:
                # Save the board image
                image_filename = f"{self.image_dir}/board_{move_number}_{color}_{model_name}.png"
                try:
                    import base64
                    from PIL import Image
                    import io
                    # Decode base64 image and save
                    image_data = base64.b64decode(board_image)
                    image = Image.open(io.BytesIO(image_data))
                    image.save(image_filename)
                    f.write(f"Board image saved: {image_filename}\n")
                except Exception as e:
                    f.write(f"Error saving board image: {str(e)}\n")
            
            f.write("\n")

    def get_move(self, model_name, fen, game_history="", move_number=None, color=None):
        """Get move from model."""
        print(f"[get_move] Called for model: {model_name}, FEN: {fen}")
        if not self.ensure_model_loaded(model_name):
            print(f"[get_move] Model loading failed for {model_name}")
            return None, "Model loading failed", ""
        
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        print(f"[get_move] Legal moves: {legal_moves}")
        if not legal_moves:
            print(f"[get_move] No legal moves available")
            return None, "No legal moves available", ""
        
        config = self.get_model_config(model_name)
        board_vis = self.generate_board_visualization(board)
        
        # Generate board image if model supports multimodal
        board_image = None
        piece_legend = "Piece letter legend: P/p = pawn, N/n = knight, B/b = bishop, R/r = rook, Q/q = queen, K/k = king (Uppercase = white, lowercase = black)"
        if config.get("supports_multimodal", False):
            board_image = self.generate_board_image(board)
            print(f"[get_move] Generated board image for multimodal model {model_name}")
        
        # Set timeout based on model type
        timeout = self.multimodal_timeout if config.get("supports_multimodal", False) else self.standard_timeout
        print(f"[get_move] Using timeout of {timeout} seconds for model {model_name}")
        
        # Improved FEN explanation
        fen_explanation = """
FEN Notation Guide:
- First part: Piece placement (e.g., 'rnbqkbnr' = rook, knight, bishop, queen, king, bishop, knight, rook)
- Second part: Active color ('w' = white to move, 'b' = black to move)
- Third part: Castling availability (KQkq = all castling available)
- Fourth part: En passant target square ('-' if none)
- Fifth part: Halfmove clock (number of halfmoves since last capture/pawn advance)
- Sixth part: Fullmove number (starts at 1, increments after black's move)
"""
        
        # Prompt improvements
        move_info = f"Move number: {move_number if move_number is not None else '?'}\nPlaying as: {color if color else '?'}\n"
        board_image_note = "The board image uses piece letters (uppercase for white, lowercase for black) and each square is labeled with its algebraic notation in the bottom left."
        forceful = f"\n\nIMPORTANT: ONLY respond with one of the following moves: {', '.join(legal_moves)}. Do NOT invent moves. If you cannot, respond with ERROR."
        
        # Add piece legend for multimodal models
        multimodal_legend = f"\n\n{piece_legend}" if config.get("supports_multimodal", False) else ""
        
        base_prompt = f"""You are a chess grandmaster. Analyze this position and make the best move.\n\n{move_info}\nCurrent position (FEN): {fen}\n{fen_explanation}\n\nBoard visualization:\n{board_vis}\n\n{board_image_note}{multimodal_legend}\n\nGame history: {game_history}\nLegal moves: {', '.join(legal_moves)}{forceful}\n\nIMPORTANT INSTRUCTIONS:\n1. You MUST choose a move from the legal moves list provided above\n2. Your response must be in this exact format:\n<thinking>\n[Your analysis and reasoning here]\n</thinking>\n\nMOVE: [move in UCI notation]\n\nFor example:\n<thinking>\nI will develop my knight to control the center.\n</thinking>\n\nMOVE: e2e4\n\nIf you cannot find a valid move, respond with:\nERROR: [explanation]"""

        if config["supports_cot"]:
            prompt = base_prompt + """\n<thinking>\nThink step by step:\n1. Analyze the current position:\n   - Which pieces are where?\n   - Who's turn is it?\n   - What are the key tactical and strategic elements?\n2. Review the legal moves provided\n3. Evaluate each promising candidate move:\n   - Does it develop pieces?\n   - Does it control the center?\n   - Does it create threats?\n   - Does it maintain king safety?\n4. Choose the best move based on your analysis\n5. Explain your reasoning\n</thinking>\n\nAfter your analysis, provide your move in UCI notation (e.g., e2e4, g1f3).\nFormat your final answer as: MOVE: [your move in UCI notation]"""
        else:
            prompt = base_prompt + """\nYour task:\n1. Analyze the position\n2. Choose the best move from the legal moves list\n3. Provide your move in UCI notation (e.g., e2e4, g1f3)\n\nFormat: MOVE: [your move in UCI notation]"""

        try:
            print(f"[get_move] Sending request to Ollama API for model {model_name}")
            response = None
            if config.get("supports_multimodal", False):
                # For multimodal models, include the image in the request
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": True,
                        "images": [board_image] if board_image else [],
                        "options": {
                            "temperature": config["temperature"],
                            "top_p": config["top_p"],
                            "num_predict": config["max_tokens"]
                        }
                    },
                    timeout=timeout
                )
            else:
                # For non-multimodal models
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": config["temperature"],
                            "top_p": config["top_p"],
                            "num_predict": config["max_tokens"]
                        }
                    },
                    timeout=timeout
                )

            if response and response.status_code == 200:
                full_thinking = ""
                chain_of_thought = ""
                print(f"[get_move] Begin streaming response lines...")
                for line in response.iter_lines():
                    print(f"[get_move] Got line: {line}")
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                token = chunk['response']
                                full_thinking += token
                                if '<thinking>' in full_thinking and '</thinking>' in full_thinking:
                                    start = full_thinking.find('<thinking>') + 11
                                    end = full_thinking.find('</thinking>')
                                    chain_of_thought = full_thinking[start:end].strip()
                                if len(full_thinking) % 20 == 0:
                                    socketio.emit('thinking_stream', {
                                        'model': model_name,
                                        'thinking': chain_of_thought if chain_of_thought else full_thinking,
                                        'complete': False
                                    })
                                if chunk.get('done', False):
                                    socketio.emit('thinking_stream', {
                                        'model': model_name,
                                        'thinking': chain_of_thought if chain_of_thought else full_thinking,
                                        'complete': True
                                    })
                                    print(f"[get_move] Streaming complete, breaking loop.")
                                    break
                        except json.JSONDecodeError as e:
                            print(f"[get_move] JSON decode error: {e} for line: {line}")
                            continue
                print(f"[get_move] Full thinking: {full_thinking}")
                move = self.extract_move(full_thinking)
                print(f"[get_move] Extracted move: {move}")
                
                # Log the move
                self.log_move(
                    move_number=move_number,
                    color=color,
                    model_name=model_name,
                    prompt=prompt,
                    response=full_thinking,
                    board_image=board_image if config.get("supports_multimodal", False) else None
                )
                
                # Emit the full model response to the frontend
                socketio.emit('model_response', {
                    'model': model_name,
                    'response': full_thinking
                })
                
                if move:
                    if move in legal_moves:
                        return move, full_thinking, chain_of_thought
                    else:
                        print(f"[get_move] Invalid move {move} not in legal moves: {legal_moves}")
                        return None, f"Invalid move: {move} not in legal moves list", chain_of_thought
                else:
                    print(f"[get_move] No valid move found in response: {full_thinking}")
                    return None, "No valid move found in response", chain_of_thought
            else:
                error_msg = f"API Error: {response.status_code if response else 'No response'}"
                print(f"[get_move] {error_msg}")
                return None, error_msg, ""
        except Exception as e:
            error_msg = f"Error in get_move: {str(e)}"
            print(f"[get_move] {error_msg}")
            return None, error_msg, ""
    
    def generate_board_visualization(self, board):
        """Generate a visual representation of the board."""
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        
        pieces = {
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
        }
        
        board_vis = "  " + " ".join(files) + "\n"
        for rank in ranks:
            board_vis += rank + " "
            for file in files:
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                if piece:
                    board_vis += pieces[piece.symbol()] + " "
                else:
                    board_vis += ". "
            board_vis += rank + "\n"
        board_vis += "  " + " ".join(files)
        
        return board_vis
    
    def extract_move(self, response):
        """Extract move from model response."""
        print(f"[extract_move] Extracting move from response: {response}")
        
        # First try to find MOVE: pattern
        move_match = re.search(r'MOVE:\s*([a-h][1-8][a-h][1-8](?:[qrbn])?)', response.lower())
        if move_match:
            move = move_match.group(1)
            print(f"[extract_move] Found move using MOVE: pattern: {move}")
            return move
        
        # If no MOVE: pattern, try to find any UCI move in the response
        uci_pattern = r'[a-h][1-8][a-h][1-8](?:[qrbn])?'
        move_match = re.search(uci_pattern, response.lower())
        if move_match:
            move = move_match.group(0)
            print(f"[extract_move] Found move using UCI pattern: {move}")
            return move
        
        print(f"[extract_move] No move found in response")
        return None

# Global instances
ollama_manager = OllamaManager()
current_game = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/available_models')
def get_available_models():
    return jsonify(MODELS)

@app.route('/api/start_game', methods=['POST'])
def start_game():
    global current_game
    
    data = request.json
    white_model = data.get('white_model')
    black_model = data.get('black_model')
    
    if not white_model or not black_model:
        return jsonify({'error': 'Both models must be specified'}), 400
        
    if white_model not in MODELS or black_model not in MODELS:
        return jsonify({'error': 'Invalid model selection'}), 400
        
    if current_game and not current_game.game_over:
        return jsonify({'error': 'A game is already in progress'}), 400
        
    current_game = ChessGame()
    
    # Start game thread
    threading.Thread(target=play_game, args=(white_model, black_model), daemon=True).start()
    
    return jsonify({'status': 'Game started'})

def play_game(white_model, black_model):
    """Play a single game between two models."""
    move_number = 1
    white_san = None
    black_san = None
    print(f"Starting play_game: white={white_model}, black={black_model}")
    
    while not current_game.game_over:
        current_model = white_model if current_game.current_player == "white" else black_model
        print(f"Current player: {current_game.current_player}, Model: {current_model}")
        print(f"Current FEN: {current_game.board.fen()}")
        
        # Get move from model
        move, thinking, chain_of_thought = ollama_manager.get_move(
            current_model,
            current_game.board.fen(),
            game_history=" ".join(current_game.move_history),
            move_number=move_number,
            color=current_game.current_player
        )
        
        print(f"Model {current_model} returned move: {move}, thinking: {thinking}")
        
        if move:
            # Validate move and get SAN
            is_valid, san, reason = ollama_manager.validate_move(move, current_game.board)
            
            if is_valid:
                # Make the move
                success, reason = current_game.make_move(move)
                print(f"Tried to make move {move}: success={success}, reason={reason}")
                
                if success:
                    if current_game.current_player == "black":
                        white_san = san
                    else:
                        black_san = san
                        move_number += 1
                    
                    # Emit move made event
                    socketio.emit('move_made', {
                        'model': current_model,
                        'move': move,
                        'san': san,
                        'move_number': move_number,
                        'white_san': white_san,
                        'black_san': black_san,
                        'fen': current_game.board.fen(),
                        'game_over': current_game.game_over,
                        'result': "1-0" if current_game.winner == "white" else "0-1" if current_game.winner == "black" else "1/2-1/2"
                    })
                else:
                    print(f"Move error for {current_model}: {reason}")
                    socketio.emit('move_error', {
                        'model': current_model,
                        'error': reason
                    })
            else:
                print(f"Invalid move {move} for {current_model}: {reason}")
                socketio.emit('move_error', {
                    'model': current_model,
                    'error': reason
                })
        else:
            print(f"No move returned for {current_model}, thinking: {thinking}")
            socketio.emit('move_error', {
                'model': current_model,
                'error': thinking
            })
    
    print("Game over. Unloading models.")
    # Unload both models when game is over
    ollama_manager.unload_current_model()

if __name__ == '__main__':
    print("Starting Chess Game Server...")
    print("Available models:", MODELS)
    socketio.run(app, host='0.0.0.0', port=8001, debug=False)