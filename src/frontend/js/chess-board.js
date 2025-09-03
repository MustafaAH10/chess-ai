/**
 * Chess board rendering and interaction functionality
 */
class ChessBoardManager {
    
    renderBoard(currentFen) {
        const board = document.getElementById('chessBoard');
        board.innerHTML = '';

        const pieces = this.fenToPieces(currentFen);

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `chess-square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.square = this.indexToSquare(row, col);
                
                const piece = pieces[row][col];
                if (piece) {
                    square.textContent = this.pieceToUnicode(piece);
                }

                square.onclick = (event) => this.onSquareClick(row, col, event);
                board.appendChild(square);
            }
        }
    }

    fenToPieces(fen) {
        const board = Array(8).fill().map(() => Array(8).fill(null));
        const position = fen.split(' ')[0];
        const rows = position.split('/');

        for (let row = 0; row < 8; row++) {
            let col = 0;
            for (let char of rows[row]) {
                if (isNaN(char)) {
                    board[row][col] = char;
                    col++;
                } else {
                    col += parseInt(char);
                }
            }
        }

        return board;
    }

    pieceToUnicode(piece) {
        const pieces = {
            'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
            'k': '\u265A', 'q': '\u265B', 'r': '\u265C', 'b': '\u265D', 'n': '\u265E', 'p': '\u265F'
        };
        return pieces[piece] || '';
    }

    indexToSquare(row, col) {
        return String.fromCharCode(97 + col) + (8 - row);
    }

    onSquareClick(row, col, event) {
        // This will be bound to the main game instance
        if (typeof window.game !== 'undefined') {
            window.game.handleSquareClick(row, col);
        }
    }
}

// Add methods to ChessGame for board interaction
ChessGame.prototype.handleSquareClick = function(row, col) {
    const square = document.querySelector(`[data-square="${this.boardManager.indexToSquare(row, col)}"]`);
    
    if (this.selectedSquare) {
        const fromSquare = this.selectedSquare;
        const toSquare = this.boardManager.indexToSquare(row, col);
        
        // Clear selection
        document.querySelectorAll('.chess-square').forEach(s => {
            s.classList.remove('selected');
        });
        this.selectedSquare = null;
        
        // Try to make move
        if (fromSquare !== toSquare) {
            this.makeMove(`${fromSquare}${toSquare}`);
        }
    } else {
        // Select square
        document.querySelectorAll('.chess-square').forEach(s => {
            s.classList.remove('selected');
        });
        square.classList.add('selected');
        this.selectedSquare = this.boardManager.indexToSquare(row, col);
    }
};

ChessGame.prototype.renderBoard = function() {
    if (!this.boardManager) {
        this.boardManager = new ChessBoardManager();
    }
    this.boardManager.renderBoard(this.currentFen);
};

ChessGame.prototype.updateGameStatus = function() {
    const statusEl = document.getElementById('gameStatus');
    const historyEl = document.getElementById('moveHistory');
    
    const isWhiteToMove = this.currentFen.includes(' w ');
    statusEl.textContent = `${isWhiteToMove ? 'White' : 'Black'} to move`;
    
    historyEl.textContent = this.gameHistory.length > 0 
        ? this.gameHistory.join(' ') 
        : 'Game started';
};