import numpy as np
from src.game.chess_env import ChessEnv
from src.mcts.mcts import MCTS


class SelfPlay:
    def __init__(self, model, num_simulations=100, temperature=1.0):
        self.mcts = MCTS(model, num_simulations=num_simulations)
        self.temperature = temperature
    
    def generate_game(self):
        env = ChessEnv()
        game_history = []
        
        while not env.is_game_over():
            state = env.get_state()
            current_player = 1 if env.board.turn == chess.WHITE else -1
            
            visit_counts = self.mcts.search(env)
            legal_moves = env.legal_moves_uci()
            policy_target = np.zeros(len(legal_moves))
            
            for i, move in enumerate(legal_moves):
                if move in visit_counts:
                    policy_target[i] = visit_counts[move]
            
            if policy_target.sum() > 0:
                policy_target /= policy_target.sum()
            else:
                policy_target = np.ones(len(legal_moves)) / len(legal_moves)
            
            game_history.append((state.copy(), policy_target.copy(), current_player))
            move = self.mcts.get_best_move(env, temperature=self.temperature)
            env.step(move)
        
        result_str = env.result()
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = -1.0
        else:
            result = 0.0
        
        training_data = []
        for state, policy, player in game_history:
            value_target = result * player
            training_data.append((state, policy, value_target))
        
        return training_data, result
    
    def generate_games(self, num_games=10):
        all_data = []
        results = []
        for i in range(num_games):
            print(f"Generando partida {i+1}/{num_games}...")
            game_data, result = self.generate_game()
            all_data.extend(game_data)
            results.append(result)
        return all_data, results
