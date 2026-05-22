import math
import numpy as np
import torch


class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
    
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, model, c_puct=1.5, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root = MCTSNode()
    
    def search(self, env):
        self.root = MCTSNode()
        for _ in range(self.num_simulations):
            env_copy = env.clone()
            self._simulate(env_copy, self.root)
        
        visits = {move: node.visit_count for move, node in self.root.children.items()}
        total = sum(visits.values())
        if total > 0:
            return {move: count / total for move, count in visits.items()}
        return {}
    
    def _simulate(self, env, node):
        path = [node]
        while node.is_expanded and node.children:
            move, node = self._select_child(node)
            env.step(move)
            path.append(node)
        
        if env.is_game_over():
            result = env.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0
        else:
            state = env.get_state()
            policy_logits, value = self.model.predict(state)
            legal_moves = env.legal_moves_uci()
            for move in legal_moves:
                prior = 1.0 / len(legal_moves)
                node.children[move] = MCTSNode(prior=prior)
            node.is_expanded = True
        
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
    
    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_node = None
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits)
        
        for move, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_node = child
        
        return best_move, best_node
    
    def get_best_move(self, env, temperature=1.0):
        visit_counts = self.search(env)
        if not visit_counts:
            moves = env.legal_moves_uci()
            return np.random.choice(moves)
        
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves], dtype=np.float32)
        
        if temperature == 0:
            return moves[np.argmax(counts)]
        
        counts = counts ** (1.0 / temperature)
        probs = counts / counts.sum()
        return np.random.choice(moves, p=probs)
