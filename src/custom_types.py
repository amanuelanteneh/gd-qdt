from dataclasses import dataclass

from optuna import Trial


@dataclass
class Hyperparameters:
    lr: float = 1e-3
    lr_decay: float = 0.999
    beta1: float = 0.999
    beta2: float = 0.99
    num_epochs: int = 2
    batch_size: int = 64
    weight_decay: float = 0.0

    @classmethod
    def from_optuna_trial(cls, trial: Trial) -> "Hyperparameters":
        lr = trial.suggest_float("lr", 1e-5, 2e-1, log=True)
        lr_decay = trial.suggest_float("lr_decay", 0.9, 0.999, log=True)
        beta1 = trial.suggest_float("beta1", 0.8, 0.999)
        beta2 = trial.suggest_float("beta2", 0.8, 0.999)
        num_epochs = trial.suggest_int("num_epochs", 25, 100, step=5)
        batch_size = trial.suggest_int("batch_size", 20, 70, step=5)

        return cls(
            lr=lr,
            lr_decay=lr_decay,
            beta1=beta1,
            beta2=beta2,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )