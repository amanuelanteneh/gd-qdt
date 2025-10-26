from math import sqrt

import torch as th
import numpy as np
import scipy.linalg as scplin


def unstack(tensor: th.Tensor, N: int, M: int) -> th.Tensor:
    """Convert a stacked tensor of shape (MN,N) to shape (M,N,N)"""
    return tensor.view(M, N, N)


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

def check_povm_validity(povm: list[th.Tensor], tol: float = 1e-6) -> bool:
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

