from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from optuna import Trial, TrialPruned

from src.loss import phase_insensitive_loss_gd
from custom_types import Hyperparameters


def learn_phase_insensitive_povm(
    logits: Tensor,
    hyperparams: Hyperparameters,
    dataset: TensorDataset,
    lam_smoothing: float,
    verbose: bool = True,
    trial: Trial = None,
    lam_l1: float = 0.0
) -> tuple[Tensor, list[float], list[float], list[int]]:
    """
    Uses gradient descent to learn a POVM that minimizes 
    Args:

    """

    optimizer = Adam(
        [logits], lr=hyperparams.lr, betas=(hyperparams.beta1, hyperparams.beta2)
    )

    # Decay LR by gamma every time called
    scheduler = ExponentialLR(optimizer, gamma=hyperparams.lr_decay)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=hyperparams.batch_size, shuffle=True)

    losses = []
    lr_vals = []
    iters = []

    epochs = (
        tqdm(range(hyperparams.num_epochs), desc="Learning POVM")
        if verbose
        else range(hyperparams.num_epochs)
    )
    
    for epoch in epochs:
        iters.append(epoch + 1)
        lr_vals.append(scheduler.get_last_lr())
        batch_losses = []

        for probes_batch, targets_batch in loader:
            optimizer.zero_grad()

            # Compute batch loss
            L = phase_insensitive_loss_gd(
                targets_batch, logits, probes_batch, lam_smoothing, lam_l1
            )

            L.backward()

            optimizer.step()

            batch_losses.append(L.item())

        scheduler.step()
        avg_batch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(avg_batch_loss)

        if trial is not None:
            trial.report(avg_batch_loss, step=epoch)
            if trial.should_prune():
                raise TrialPruned()

    return logits, losses, lr_vals, iters
