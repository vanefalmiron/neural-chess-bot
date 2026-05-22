import chess
import numpy as np

# AlphaZero codifica 8×8×73 = 4672 movimientos posibles
# 73 = 56 underpromotions + 8 knight + 8 bishop + 8 rook + 8 queen + 9 king (incluyendo castle)

# Simplificación: usaremos un mapeo UCI → índice dinámico
# En producción implementarías la codificación exacta de AlphaZero

class MoveEncoder:
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_mapping()
    
    def _build_mapping(self):
        """Pre-calcula todos los movimientos posibles"""
        board = chess.Board()
        # Generamos movimientos desde posiciones típicas para cubrir casos
        # (simplificado - en producción cubrir todos los casos)
        
        idx = 0
        # Movimientos simples de peones, caballos, etc.
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                move = chess.Move(from_sq, to_sq)
                if move in board.legal_moves:
                    uci = move.uci()
                    self.move_to_index[uci] = idx
                    self.index_to_move[idx] = uci
                    idx += 1
        
        self.num_actions = idx
        print(f"Total movimientos mapeados: {self.num_actions}")
    
    def encode(self, move_uci, legal_moves):
        """Convierte UCI a índice (solo entre movimientos legales actuales)"""
        legal_indices = []
        for move in legal_moves:
            uci = move.uci()
            if uci in self.move_to_index:
                legal_indices.append(self.move_to_index[uci])
        return legal_indices
    
    def decode(self, action_idx):
        """Índice a UCI"""
        return self.index_to_move.get(action_idx, None)

# Instancia global
encoder = MoveEncoder()