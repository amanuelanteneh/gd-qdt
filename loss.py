import torch as th
from torch import Tensor

from utils import unstack

def povm_loss(targets: Tensor, factors: Tensor, probes: Tensor, lam: float) -> Tensor:
    """
    Differentiable loss for POVM optimization (vectorized).

    Args:
        targets: (num_probes, M) real tensor — target probabilities.
        probes: (num_probes, N) complex tensor — probe state amplitudes.
        factors: (M*N, N) complex tensor — stacked POVM factors.
        lam: regularization parameter.

    Returns:
        Differentiable scalar loss tensor
    """

    N = probes.shape[1]
    M = targets.shape[1]

    # Reshape factors into (M, N, N)
    factors = unstack(factors, N=N, M=M)

    # Compute POVM elements E_m = A_m† A_m → shape (M, N, N)
    povm = th.matmul(factors.conj().transpose(-1, -2), factors)

    # Compute predicted probabilities:
    # p_i(m) = <ψ_i| E_m |ψ_i>
    # Einsum pattern explanation:
    #   b = batch (probe index)
    #   m = POVM element index
    #   i,j = Hilbert indices
    pred_probs = th.einsum("bi,mij,bj->bm", probes.conj(), povm, probes).real

    # Squared error loss
    sq_err = th.sum((pred_probs - targets) ** 2)
    #sq_err = th.linalg.matrix_norm(pred_probs - targets, ord='fro')
    
    # L1 regularization on povm elements
    reg = th.sum(th.abs(povm))

    # Total loss (negative for maximization if needed)
    return (sq_err + lam * reg)