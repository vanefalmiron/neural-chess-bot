# test_game.py
from src.game.chess_env import ChessEnv

env = ChessEnv()
state = env.reset()
print("Shape:", state.shape)  # (17, 8, 8)
print("Legal moves:", len(env.legal_moves()))  # 20 (apertura)

# Juega e4
state, reward, done = env.step("e2e4")
print("After e4:", len(env.legal_moves()))  # 20 respuestas