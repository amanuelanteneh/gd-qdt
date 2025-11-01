import torch as th
from torch import Tensor
from torch.utils.data import TensorDataset
from optuna import Trial

from gd_opt import learn_phase_insensitive_povm
from custom_types import Hyperparameters


def objective(
    trial: Trial,
    probes: Tensor,
    targets: Tensor,
    lam_smoothing: float,
    device: str,
) -> float:
    """
    Objective function for hyperparameter tuning using Optuna.

    Args:
        trial: The Optuna trial object.
        dataset: The observed dataset.
        device: The device to use for training (e.g., 'cuda' or 'cpu').

    Returns:
        float: The objective metric to minimize.
    """

    M, N, = probes.shape[1], targets.shape[1]
    hyperparams = Hyperparameters.from_optuna_trial(trial)

    # weight initialization
    logits = th.rand((M, N)) - 1
    logits = logits.to(device)
    logits.requires_grad = True  # tell Torch to track gradients for `logits` 
    
    probes = probes.to(device)
    targets = targets.to(device)
    dataset = TensorDataset(probes, targets)

    logits, losses, lr_vals, iters = learn_phase_insensitive_povm(logits, hyperparams, dataset, lam_smoothing)

    trial.set_user_attr("logits", logits)
    trial.set_user_attr("losses", losses)
    trial.set_user_attr("lr_vals", lr_vals)
    trial.set_user_attr("iters", iters)

    return losses[-1]