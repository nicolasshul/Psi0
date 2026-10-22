import torch
from torch.utils.data import Dataset

class Standard_Scaler:
    ''''Z-score'''
    def fit(self, data):
        self.mean = data.mean(dim = 0)
        self.std = data.std(dim = 0)

        self.std = torch.clamp(self.std, min=1e-6)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

class ResidualDataset(Dataset):

    def __init__(self, observations, psi_actions, expert_actions, offset=6):
        self.observations = observations
        self.psi_actions = psi_actions
        self.expert_actions = expert_actions

        self.offset = offset

        self.length = min(
            len(psi_actions),
            len(observations) - offset,
            len(expert_actions) - offset
        )

    def __len__(self):
        return self.length

    def __getitem__(self, t):

        # Psi0 action from the original / older timestep
        psi_action = self.psi_actions[t]

        # Observation offset into the future
        current_obs = self.observations[t + self.offset]

        # Expert action at the SAME timestep as current_obs
        expert_action = self.expert_actions[t + self.offset]

        # Input to residual MLP
        x = torch.cat(
            [
                current_obs,
                psi_action
            ],
            dim=-1
        )

        return {
            "input": x,
            "psi_action": psi_action,
            "expert_action": expert_action
        }
