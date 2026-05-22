import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels=17, out_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PolicyHead(nn.Module):
    def __init__(self, channels=128, num_actions=4672):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, num_actions)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ValueHead(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))


class ChessResNet(nn.Module):
    def __init__(self, num_blocks=10, channels=128, num_actions=4672):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        
        self.conv_block = ConvBlock(17, channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        self.policy_head = PolicyHead(channels, num_actions)
        self.value_head = ValueHead(channels)
    
    def forward(self, x):
        out = self.conv_block(x)
        for block in self.res_blocks:
            out = block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value
    
    def predict(self, state):
        self.eval()
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state = state.float()
            else:
                state = torch.from_numpy(state).float()
            if state.dim() == 3:
                state = state.unsqueeze(0)
            policy, value = self.forward(state)
            return policy.squeeze(0).numpy(), value.item()
