import chess
import numpy as np

class ChessEnv:
    """
    Wrapper sobre python-chess para AlphaZero.
    """
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        self.board = chess.Board()
        return self.get_state()
    
    def get_state(self):
        """
        Convierte tablero a tensor 8x8x17
        """
        state = np.zeros((17, 8, 8), dtype=np.float32)
        
        # 12 planos: piezas (6 tipos × 2 colores)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8  # Flip filas
                plane = piece_map[piece.symbol()]
                state[plane, row, col] = 1
        
        # Planes adicionales (simplificado por ahora)
        state[12] = 1 if self.board.turn == chess.WHITE else 0  # Turno
        
        return state
    
    def legal_moves(self):
        return list(self.board.legal_moves)
    
    def step(self, move_uci):
        """
        Ejecuta movimiento, retorna (next_state, reward, done)
        """
        move = chess.Move.from_uci(move_uci)
        self.board.push(move)
        
        done = self.board.is_game_over()
        reward = 0
        
        if done:
            result = self.board.result()
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            # "1/2-1/2" = 0
        
        return self.get_state(), reward, done
    
    def legal_moves_uci(self):
        """Retorna lista de movimientos legales en formato UCI."""
        return [move.uci() for move in self.board.legal_moves]
    
    def clone(self):
        """Para MCTS"""
        env = ChessEnv()
        env.board = self.board.copy()
        return env
    
    def is_game_over(self):
        return self.board.is_game_over()
    
    def result(self):
        return self.board.result()