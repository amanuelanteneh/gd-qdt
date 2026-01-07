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


def unstack(tensor: th.Tensor, N: int, M: int) -> th.Tensor:
    """Convert a stacked tensor of shape (MN,N) to shape (M,N,N)"""
    return tensor.view(M, N, N)


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

def circle_points(N: int, R: float = 1.0):
    """
    Return N evenly spaced points on a circle of radius R in the complex plane.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = R * np.exp(1j * angles)
    return points


def grid_points(n: int, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), dtype=th.complex128):
    """
    Generate an n x n grid of complex points in the complex plane.
    
    Args:
        n: number of points along each axis
        xlim: tuple (xmin, xmax)
        ylim: tuple (ymin, ymax)
        dtype: complex dtype
    
    Returns:
        Tensor of shape (n*n,) containing complex points
    """
    # Create evenly spaced coordinates
    re = th.linspace(xlim[0], xlim[1], n)
    im = th.linspace(ylim[0], ylim[1], n)
    Re, Im = th.meshgrid(re, im, indexing="xy")

    # Combine into complex numbers
    Z = Re + 1j * Im
    return Z.reshape(-1).to(dtype)


def check_povm(povm: list[th.Tensor], tol: float = 1e-6) -> bool:
    """
    Check if a given POVM is valid.

    Args:
        povm: List of POVM elements (tensors).
        tol: Tolerance for numerical checks.    
    Returns:
        bool: True if the POVM is valid, False otherwise. 
    """

    # Check positivity
    for E in povm:
        eigenvalues = th.linalg.eigvalsh(E)
        if th.any(eigenvalues < -tol):
            print("One or more POVM elements is not positive semi-definite.")
            return False

    # Check completeness
    identity = th.eye(povm[0].shape[0], dtype=povm[0].dtype)
    sum_E = sum(povm)
    if th.linalg.norm(sum_E - identity, ord=2) > tol:
        print("POVM elements do not sum to identity.")
        return False

    return True


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

def on_stiefel(mat: th.Tensor, tol: float = 1e-6) -> bool:
    """
    Check if a given matrix is on the Stiefel manifold

    Args:
        mat: nxp matrix
        tol: Tolerance for numerical checks.    
    Returns:
        bool: True if the POVM is valid, False otherwise. 
    """
    # Check orthonormality
    identity = th.eye(mat.shape[1], dtype=mat[0].dtype)
    if th.linalg.norm(mat.H @ mat - identity, ord=2) > tol:
        print("Matrix is not orthonormal")
        return False

    return True

def random_stiefel(rows: int, cols: int, dtype: th.dtype = th.float64) -> th.Tensor:
    """
    Generates a random matrix on the Stiefel manifold 
    St(rows, cols) i.e. a random complex matrix with orthonormal columns.

    Args:
        rows: The number of rows in the matrix.
        cols: The number of columns in the matrix.

    Returns:
        Tensor: A complex matrix with orthonormal columns.
    """
    # Generate a random complex matrix
    # Real and imaginary parts are drawn from a standard normal distribution
    random_real = th.randn(rows, cols, dtype=dtype)
    random_imag = th.randn(rows, cols, dtype=dtype)
    random_complex_matrix = th.complex(random_real, random_imag)

    # Perform QR decomposition
    # The 'Q' matrix will have orthonormal columns
    Q, _ = th.linalg.qr(random_complex_matrix)

    return Q

def random_povm(N: int, M: int) -> list[th.Tensor]:
    """
    Generate a random POVM {E_i} on an N-dimensional Hilbert space
    with M outcomes.
    """

    # Step 1: random positive semidefinite operators
    Fs = []
    for _ in range(M):
        real_part = th.randn(N, N)
        imag_part = th.randn(N, N)
        A = (real_part + 1j * imag_part) / sqrt(2.0)
        F = (A.H @ A).to(dtype=th.complex128)  # Hermitian, PSD
        Fs.append(F)

    # Step 2: normalize so they sum to identity
    S = sum(Fs)

    # matrix square root inverse
    sqrtS = th.tensor(scplin.sqrtm(S).astype(np.complex128))
    S_inv_sqrt = th.linalg.inv(sqrtS).to(dtype=th.complex128)
    Es = [S_inv_sqrt @ F @ S_inv_sqrt for F in Fs]

    return Es

def plot_matrix(
    matrix,
    title="Matrix Plot",
    cmap="viridis",
    show_values=False,
    xlabel="Columns",
    ylabel="Rows",
    colorbar=True,
    figsize=(6, 5),
    vmin=None,
    vmax=None
):
    """
    Plots a 2D matrix using matplotlib.

    Args:
        matrix (np.ndarray): 2D matrix to plot.
        title (str): Title of the plot.
        cmap (str): Colormap (e.g. 'viridis', 'plasma', 'RdBu', 'gray').
        show_values (bool): If True, display values inside cells.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        colorbar (bool): Whether to display colorbar.
        figsize (tuple): Figure size in inches.
        vmin, vmax: Optional limits for color scale.
    """
    matrix = np.array(matrix)

    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)

    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)

    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center", color="white" if abs(matrix[i, j]) > (np.max(matrix)/2) else "black")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

