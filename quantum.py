from math import gamma, sqrt, exp, comb
import torch as th
from torch import Tensor, tensor, linalg, dtype, complex128, outer, diag, zeros, tensor, trace
from torch.special import gammaln
from qutip import fidelity, Qobj

def coherent_ket(
    alpha: complex, dim: int, dtype: dtype = complex128
) -> Tensor:
    """
    Returns the ket of the coherent 
    state (displaced vacuum state): D(α)|0⟩
    NOTE: we use gammaln to avoid overflow errors
    """

    const = exp(-(abs(alpha) ** 2) / 2.0)

    # old code that doesn't take into account large values of n!
    # ket = tensor(
    #     [const * (alpha ** (n)) / sqrt(gamma(n + 1)) for n in range(dim)], dtype=dtype
    # )

    ket = tensor(
        [const * (alpha ** n) * exp(-0.5 * gammaln(tensor(n + 1, dtype=th.float64))) 
         for n in range(dim)],
        dtype=dtype,
    )
    norm = linalg.norm(ket)
    ket /= norm  # normalize ket

    return ket


def coherent_dm(
    alpha: complex, dim: int, dtype: dtype = complex128
) -> Tensor:
    """
    Returns the density matrix of the coherent 
    state (displaced vacuum state): D(α)|0⟩⟨0|D(α)†
    Note that math.gamma(n) computes (n-1)! not n! so be careful.
    """

    ket = coherent_ket(alpha, dim, dtype=dtype)
    dm = outer(ket, ket.conj())

    return dm


def lossy_pnr_povm(hilbert_dim: int, eta: float, device=None, dtype=th.float64) -> list[Tensor]:
    """
    Returns the POVM of a PNR detector with quantum efficiency `eta`
    with a Fock space truncation of `hilbert_dim` constructed according to the 
    formula in Millers thesis. Note however that the inner loop starts not at
    0 but m 

    Args:
        hilbert_dim: Hilbert space cutoff (max photon number)
        eta: detection efficiency (0 <= eta <= 1)
        device, dtype: optional torch settings

    Returns:
        povms: list of N tensors [Π_0, Π_1, ..., Π_{N-1}], each (hilbert_dim, hilbert_dim), diagonal.
    """
    povms = []

    # Initialize each Π_k as a diagonal tensor
    povms = [zeros((hilbert_dim, hilbert_dim), device=device, dtype=dtype) for _ in range(hilbert_dim)]

    # Loop over input photon number n (outer loop)
    for n in range(hilbert_dim):
        # Loop over detected photon number k (inner loop)
        for k in range(n + 1):
            p = comb(n, k) * (eta ** k) * ((1 - eta) ** (n - k))
            povms[k][n, n] = p  # Diagonal element ⟨n|Π_k|n⟩ = P(k|n)
    
    return povms


def povm_fidelity(povm_a: Tensor, povm_b: Tensor) -> float:
    
    F = fidelity(Qobj(povm_a), Qobj(povm_b))**2 
    F /= (trace(povm_a)*trace(povm_b))
    return F.real.item()