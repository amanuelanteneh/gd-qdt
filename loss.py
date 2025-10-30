import torch as th
from torch import Tensor, softmax

from utils import unstack

def povm_loss(targets: Tensor, factors: Tensor, probes: Tensor, lam: float) -> Tensor:
    """
    Differentiable loss for POVM optimization (vectorized).

    Args:
        targets: (num_probes, M) real tensor — target probabilities.
        probes: (num_probes, N) complex tensor — probe state amplitudes.
        factors: (M*N, N) complex tensor — stacked POVM factors A_i such that A_i†A_i = E_i.
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
    #sq_err = th.linalg.matrix_norm(pred_probs - targets, ord='nuc')
    
    # L1 regularization on povm elements
    reg = th.linalg.norm(povm.view((M*N, N)), ord=1) #th.sum(th.abs(povm))

    # Total loss (negative for maximization if needed)
    return (sq_err + lam * reg)


def phase_insensitive_povm_loss(targets: Tensor, logits: Tensor, probes: Tensor, lam: float):
    """
    Differentiable loss for POVM optimization (vectorized).

    Args:
        targets: (num_probes, M) real tensor — target probabilities (P).
        probes: (num_probes, N) real tensor — probe state probabilities in Fock basis (F).
        logits: (M, N) real tensor. The rows correspond to diagonal elements of POVM element E_i (Π).
                POVM is diagonal in the Fock basis since we are looking at phase insensitive detector
        lam: regularization parameter.

    Returns:
        Differentiable scalar loss tensor
    """

    # Compute POVM elements
    povm_probs = th.softmax(logits, dim=1).to(th.float64)  # rows are probability vectors
    # x2 = logits ** 2
    # norm = x2.sum(dim=0, keepdim=True).to(dtype=th.complex128) + 1e-12
    # povm_probs = x2 / norm
    # povm = th.stack([ th.diag(povm_probs[i, :]) for i in range(M) ])  # row i is the diagonal of povm element E_i

    # Compute predicted probabilities:
    #pred_probs = th.einsum("bi,mij,bj->bm", probes.conj(), povm, probes).real
    pred_probs = probes @ povm_probs

    # Squared error loss
    sq_err = th.sum((pred_probs - targets) ** 2)
    #sq_err = th.linalg.matrix_norm(pred_probs - targets, ord='fro')
    #sq_err = th.linalg.matrix_norm(pred_probs - targets, ord='nuc')
    
    # L1 regularization on povm elements
    #reg = th.linalg.norm(povm.view((M*N, N)), ord=1) #th.sum(th.abs(povm))
    reg = th.sum((povm_probs[:-1, :] - povm_probs[1:, :]) ** 2).real
    #reg += th.sum(th.abs(povm_probs))  # L1 regularization

    # Total loss (negative for maximization if needed)
    return (sq_err + lam * reg)