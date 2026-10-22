import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from model import Residual_Network
from dataset import Residual_Dataset, Standard_Scaler
from loss import residual_loss


def train(model, train_loader, optimizer, device, num_epochs, lambda_residual):

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            x = batch["input"].to(device)

            psi_action = batch["psi_action"].to(device)

            expert_action = batch["expert_action"].to(device)

            optimizer.zero_grad()

            residual_pred = model(x)

            loss, bc_loss, reg_loss = (residual_loss(residual_pred, psi_action, expert_action, lambda_residual))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (total_loss / len(train_loader))

        print(f"Epoch {epoch + 1:03d} | "f"Loss: {avg_loss:.6f}")


def main():

    directory = Path(__file__).resolve().parent

    checkpoint_path = directory / "residual_policy.pt"

    if checkpoint_path.is_file():
        print("Model has already been trained")
    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Device:", device)

        input_dim = 256       # placeholder
        action_dim = 36       # placeholder

        batch_size = 64
        learning_rate = 1e-4
        num_epochs = 100
        lambda_residual = 1e-3

        N = 1000

        # Generate some data

        inputs = torch.randn(N, input_dim)

        psi_actions = torch.randn(N, action_dim)

        expert_actions = torch.randn(N, action_dim)


        # normalize input

        scaler = Standard_Scaler()

        scaler.fit(inputs)

        inputs = scaler.transform(inputs)


        # Dataset

        dataset = Residual_Dataset(inputs, psi_actions, expert_actions)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # model

        model = Residual_Network(input_dim=input_dim, action_dim=action_dim).to(device)


        # optimizer

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)


        # train

        
        train(model, train_loader, optimizer, device, num_epochs, lambda_residual)

        # save model

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_mean":
                    scaler.mean,

                "input_std":
                    scaler.std,

                "input_dim":
                    input_dim,

                "action_dim":
                    action_dim
            },
            checkpoint_path
        )

        print(
            "Saved residual_policy.pt"
        )


if __name__ == "__main__":
    main()