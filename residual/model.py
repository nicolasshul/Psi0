import torch.nn as nn

class Residual_Network(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),

            nn.Linear(256, 128), nn.ReLU(),

            nn.Linear(128, 64), nn.ReLU(),

            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)