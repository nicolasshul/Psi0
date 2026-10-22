import torch
import torch.nn.functional as F


def residual_loss(residual_pred, psi_action, expert_action, lambda_residual=1e-3):
    corrected_action = (psi_action + residual_pred)

    bc_loss = F.mse_loss(corrected_action, expert_action)

    residual_reg = torch.mean(residual_pred ** 2)

    total_loss = (bc_loss + lambda_residual * residual_reg)

    return (total_loss, bc_loss, residual_reg)