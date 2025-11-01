from tqdm import tqdm
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from loss import phase_insensitive_loss_gd
from custom_types import Hyperparameters


def learn_phase_insensitive_povm(
    logits: Tensor,
    hyperparams: Hyperparameters,
    dataset: TensorDataset,
    lam_smoothing: float,
) -> tuple[Tensor, list[float], list[float], list[int]]:
    """
    Uses gradient descent to
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

    for iter in tqdm(range(hyperparams.num_epochs), desc="Learning POVM"):
        
        iters.append(iter + 1)
        lr_vals.append(scheduler.get_last_lr())
        batch_losses = []

        for probes_batch, targets_batch in loader:
            optimizer.zero_grad()

            # Compute batch loss
            L = phase_insensitive_loss_gd(
                targets_batch, logits, probes_batch, lam_smoothing
            )

            L.backward()

            optimizer.step()

            batch_losses.append(L.item())

        scheduler.step()
        losses.append(sum(batch_losses) / len(batch_losses))

    return logits, losses, lr_vals, iters
