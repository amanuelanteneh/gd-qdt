import numpy as np
import cvxpy as cp
import torch as th
from torch import Tensor

from utils import unstack, square_normalize


def phase_insensitive_loss_gd(
    targets: Tensor, logits: Tensor, probes: Tensor, lam_smoothing: float, lam_l1: float = 0.0, prob_norm_fn: str = "smax"
):
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
    if prob_norm_fn == "sqr":
        Pi = square_normalize(logits) # rows are probability vectors
    elif prob_norm_fn == "smax":
        Pi = th.softmax(logits, dim=1) # rows are probability vectors
    else:
        raise ValueError("The value provided is not a valid normalization option.")

    # Compute predicted probabilities:
    pred_probs = probes @ Pi

    # Squared error loss
    sq_err = th.sum((pred_probs - targets) ** 2)

    # # MLE error loss
    # eps = 1e-12
    # sq_err = -th.sum(targets * th.log(pred_probs + eps))

    # Regularization term (smoothness across consecutive POVM elements)
    reg = lam_smoothing * th.sum((Pi[:-1, :] - Pi[1:, :]) ** 2)
    
    # LASSO (L1) regularization term
    reg += lam_l1 * th.sum(th.abs(Pi))  # L1 regularization

    # Total loss
    return sq_err +  reg


def phase_insensitive_loss_cvx(
    targets: np.ndarray, probes: np.ndarray, lam_smooth: float = 0.1, solver: str = "MOSEK", lam_l1: float = 0.0
) -> tuple[np.ndarray, float, int]:
    """
    Solve convex optimization of phase-insensitive POVM loss using CVXPY.
    D = num_probes, N = number of POVM elements (outcomes), M = Hilbert dim

    Args:
        targets: (D, N) real numpy array — target probabilities
        probes: (D, M) real numpy array — probe state Fock probabilities
        lam: regularization coefficient
    Returns:
        Problem solution, solution loss value, number of iterations to converge
    """

    _, M = probes.shape
    N = targets.shape[1]

    # Variables: POVM diagonals (each column = diagonal of one POVM element)
    Pi = cp.Variable((M, N), nonneg=True)

    # Normalization constraint: sum over all POVM elements = I (each ROW (axis=1) sums to 1)
    constraints = [cp.sum(Pi, axis=1) == 1]

    # Predicted probabilities: p[m, b] = |ψ_b|^2 ⋅ Π_m
    pred_probs = probes @ Pi  # shape (D, M)

    sq_err = cp.sum_squares(pred_probs - targets)

    # Regularization term (smoothness across consecutive POVM elements)
    reg = lam_smooth * cp.sum_squares(Pi[:-1, :] - Pi[1:, :]) 
    # L1 regularization
    reg += lam_l1 *  cp.sum(cp.norm1(Pi))  

    objective = cp.Minimize(sq_err + reg)

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver, verbose=False)

    return Pi.value, problem.value, problem.solver_stats.num_iters


def phase_sensitive_loss_gd(
    targets: Tensor, factors: Tensor, probes: Tensor, lam: float
) -> Tensor:
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
    U = probes.shape[0]
    #targets = targets.T
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
    # rows = []
    # for i in range(M):
    #     row = [th.trace(povm[i] @ probes[j]).real for j in range(U)]
    #     rows.append(th.hstack(row))
    
    # pred_probs = th.vstack(rows)

    # Squared error loss
    sq_err = th.sum((pred_probs - targets) ** 2)

    # L1 regularization on povm elements
    reg = lam * th.sum(th.abs(povm))

    # Total loss (negative for maximization if needed)
    return sq_err + reg