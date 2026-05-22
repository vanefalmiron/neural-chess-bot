import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(policy).float(),
            torch.tensor([value], dtype=torch.float32)
        )


class Trainer:
    def __init__(self, model, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_loss_fn = nn.MSELoss()
    
    def train_step(self, states, policy_targets, value_targets):
        self.model.train()
        self.optimizer.zero_grad()
        
        policy_logits, values = self.model(states)
        
        value_loss = self.value_loss_fn(values, value_targets)
        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets * policy_log_probs).sum(dim=1).mean()
        
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def train_epoch(self, data, batch_size=32):
        dataset = ChessDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        total_v_loss = 0
        total_p_loss = 0
        num_batches = 0
        
        for states, policies, values in loader:
            metrics = self.train_step(states, policies, values)
            total_loss += metrics['loss']
            total_v_loss += metrics['value_loss']
            total_p_loss += metrics['policy_loss']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'value_loss': total_v_loss / num_batches,
            'policy_loss': total_p_loss / num_batches
        }
    
    def save_checkpoint(self, path, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint guardado: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['iteration']
