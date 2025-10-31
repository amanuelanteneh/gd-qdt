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


def lossy_pnr_povm(hilbert_dim: int, N: int, eta: float, dtype=th.float64) -> list[th.Tensor]:
    """
    Returns the POVM of a lossy PNR detector (up to N detected photons)
    with quantum efficiency `eta` and a Fock-space truncation of `hilbert_dim`.

    Based on Miller's thesis formula:
        ⟨n|Π_k|n⟩ = C(n, k) * η^k * (1 - η)^(n - k)
    for 0 <= k <= min(n, N).

    Args:
        hilbert_dim: Hilbert space cutoff (max photon number considered).
        N: maximum number of detected photons (PNR resolution).
        eta: detection efficiency (0 ≤ eta ≤ 1).
        device, dtype: optional torch settings.

    Returns:
        povms: list of N+1 tensors [Π₀, Π₁, ..., Π_N], each (hilbert_dim, hilbert_dim), diagonal.
    """
    povms = [th.zeros((hilbert_dim, hilbert_dim), dtype=dtype) for _ in range(N + 1)]

    # Outer loop over input photon number n
    for n in range(hilbert_dim):
        # Inner loop over detected photon number k
        for k in range(min(n, N) + 1):
            p = comb(n, k) * (eta ** k) * ((1 - eta) ** (n - k))
            povms[k][n, n] = p  # ⟨n|Π_k|n⟩ = P(k|n)

    # Optionally, merge higher counts into Π_N (saturation)
    if hilbert_dim > N:
        for n in range(N + 1, hilbert_dim):
            p = 1 - sum(comb(n, k) * (eta ** k) * ((1 - eta) ** (n - k)) for k in range(N))
            povms[N][n, n] = p

    return povms



def povm_fidelity(povm_a: Tensor, povm_b: Tensor) -> float:
    
    F = fidelity(Qobj(povm_a), Qobj(povm_b))**2 
    F /= (trace(povm_a)*trace(povm_b))
    return F.real.item()