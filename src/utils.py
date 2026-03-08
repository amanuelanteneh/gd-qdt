from math import sqrt

import matplotlib.pyplot as plt
import torch as th
import numpy as np
import scipy.linalg as scplin
from scipy.stats import poisson


def square_normalize(x: th.Tensor, eps: float = 1e-12) -> th.Tensor:
    """
    Square each entry and normalize rows so each row sums to 1.

    Args:
        x: Tensor of shape (M, N)
        eps: Small constant for numerical stability

    Returns:
        Tensor of shape (M, N), where each row is a probability vector
    """
    # x2 = x.pow(2)
    # row_sums = x2.sum(dim=1, keepdim=True)
    # return x2 / (row_sums + eps)
    x_abs = th.abs(x)
    row_sums = x_abs.sum(dim=1, keepdim=True)
    return x_abs / (row_sums + eps)



def find_lambda_max(M: int, eps: float = 1e-5, iters: int = 100) -> float:
    """
    Find the largest lambda (coherent state average photon number) such that 
    1 - F(M-1; lambda) <= eps, where F is the Poisson CDF.

    Args:
        M: Fock-space cutoff (integer)
        eps: tolerance (float)

    Returns:
        lambda_max (float)
    """
    low, hi = 0.0, float(M-1) 

    for _ in range(iters):
        mid = 0.5 * (low + hi)
        tail = 1 - poisson.cdf(M, mid)
        if tail > eps:
            hi = mid
        else:
            low = mid

    return low

def find_M_given_lambda(lambda_avg: float, prob_threshold: float = 1e-5):
    """
    Find the smallest M (Fock-space cutoff) such that 
    1 - F(M-1; lambda_avg) <= prob_threshold, where F is the Poisson CDF.
    This gives the cutoff M for which the probability of measuring more than M-1 photons
    is below the specified threshold.
    Similar to what was done in arXiv:2306.12622.

    Args:
        lambda_avg: average photon number (float)
        prob_threshold: tolerance (float)

    Returns:
        M (int)
    """
    M = 1
    while True:
        tail = 1 - poisson.cdf(M-1, lambda_avg)
        if tail <= prob_threshold:
            break
        M += 1

    return M

def find_lambda_given_N(N: int, prob_threshold: float = 0.9, iters: int = 100):
    """
    Find the largest lambda (coherent state average photon number) such that 
    1 - F(N; lambda) > prob_threshold, where F is the Poisson CDF.
    This gives the lambda for which the probability of measuring more than N photons
    is above the specified threshold.
    Similar to what was done in arXiv:2306.12622.
    """

    low, hi = 0.0, float(2*N) 

    for _ in range(iters):
        mid = 0.5 * (low + hi)
        tail = 1 - poisson.cdf(N, mid)
        if tail > prob_threshold:
            hi = mid
        else:
            low = mid

    return low


def check_diag_povm(povm: list[th.Tensor], tol: float = 1e-6) -> bool:
    """
    Check if a given (diagonal) POVM is valid. For a diagonal matrix 
    the diagonals are the eigenvalues so we will exploit this fact to avoid
    constructing the full dense matrix.

    Args:
        povm: List of POVM (diagonal) elements (tensors).
        tol: Tolerance for numerical checks.    
    Returns:
        bool: True if the POVM is valid, False otherwise. 
    """

    # Check positivity
    for E_diag in povm:
        if th.any(E_diag < -tol):
            print("One or more POVM elements is not positive semi-definite.")
            return False

    # Check completeness
    identity = th.ones(povm[0].shape[0]).to(povm[0].device)
    sum_E = sum(povm)
    err = th.linalg.norm(sum_E - identity, ord=2)
    if err > tol:
        print(f"WARNING: Error on |I - ΣE_i|^2 is: {err}")
        return False

    return True

