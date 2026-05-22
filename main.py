#!/usr/bin/env python3
"""
Punto de entrada del motor de ajedrez AlphaZero.
"""

import torch
import numpy as np
from src.game.chess_env import ChessEnv
from src.neural_net.resnet import ChessResNet
from src.mcts.mcts import MCTS
from src.self_play.self_play import SelfPlay
from src.train import Trainer


def test_game():
    """Prueba basica de la logica del juego."""
    print("=== Test: Logica del Juego ===")
    env = ChessEnv()
    state = env.reset()
    print(f"Estado inicial shape: {state.shape}")
    print(f"Movimientos legales: {len(env.legal_moves())}")
    
    # Jugar e4
    state, reward, done = env.step("e2e4")
    print(f"Despues de e4 - Reward: {reward}, Done: {done}")
    print(f"Movimientos legales: {len(env.legal_moves())}")
    print()


def test_neural_net():
    """Prueba de la red neuronal."""
    print("=== Test: Red Neuronal ===")
    model = ChessResNet(num_blocks=5, channels=64)
    
    # Estado de prueba
    env = ChessEnv()
    state = env.get_state()
    
    policy, value = model.predict(state)
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value:.4f}")
    print(f"Parametros totales: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_mcts():
    """Prueba de MCTS."""
    print("=== Test: MCTS ===")
    model = ChessResNet(num_blocks=5, channels=64)
    mcts = MCTS(model, num_simulations=10)
    
    env = ChessEnv()
    env.reset()
    
    best_move = mcts.get_best_move(env, temperature=0)
    print(f"Mejor movimiento (10 sims): {best_move}")
    print()


def main():
    print("Chess AlphaZero - Motor de Ajedrez con IA")
    print("=" * 50)
    print()
    
    # Tests basicos
    test_game()
    test_neural_net()
    test_mcts()
    
    print("Tests completados!")
    print()
    print("Proximos pasos:")
    print("1. Implementar entrenamiento completo en train.py")
    print("2. Generar partidas de self-play en bucle")
    print("3. Evaluar modelo vs versiones anteriores")
    print("4. Exportar a formato UCI para jugar contra otros motores")


if __name__ == "__main__":
    main()