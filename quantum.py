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


def pnr_povm(hilbert_dim: int, N: int, eta: float) -> list[th.Tensor]:
    """
    Returns the POVM of a PNR detector (up to N detected photons) with efficiency η
    with quantum efficiency `eta` and a Fock-space truncation of `hilbert_dim`.
    See Millers thesis section 6.1 `Detector POVMs` for more.

    if 0 < n < N-2:
        E_n = sum_{k=0}^{M-1} (k choose n) * η^n * (1 - η)^(k - n) * |k><k|
    if n = N-1:
        E_N = I_M - sum_{k=0}^{N-2} E_k

    Args:
        hilbert_dim: Hilbert space cutoff (max photon number considered).
        N: maximum number of detected photons (PNR resolution).
        eta: detection efficiency (0 ≤ eta ≤ 1).
    Returns:
        povms: list of N+1 tensors [Π₀, Π₁, ..., Π_N], each (hilbert_dim, hilbert_dim), diagonal.
    """
    
    assert hilbert_dim >= N, "Hilbert space truncation must be at least as large as detector resolution"
    
    povm = []
    
    for n in range(0, N):
        En = th.zeros((hilbert_dim, hilbert_dim))
        for k in range(n, hilbert_dim):
            fock_proj = th.zeros((hilbert_dim, hilbert_dim))
            fock_proj[k,k] = 1.0
            P_n_given_k = comb(k, n) * (eta**n) * ((1.0-eta)**(k-n))
            En += P_n_given_k * fock_proj

        povm.append(En)

    povm.append( th.eye(hilbert_dim) - sum(povm) )

    return povm    
    

def photodetector_povm(hilbert_dim: int, eta: float, dtype=th.float64) -> list[th.Tensor]:
    """
    Returns the POVM elements of a (lossy) on–off photodetector with efficiency `eta`.
    With probability ∑ₙ (1 - η)ⁿ the detector doesn't click when there are photons present.

    The model follows:
        Π₀ = ∑ₙ (1 - η)ⁿ |n⟩⟨n|     # no-click (or missed detection)
        Π₁ = I - Π₀                 # click (at least one photon detected)

    Args:
        hilbert_dim: Hilbert space cutoff (max photon number)
        eta: detection efficiency (0 ≤ η ≤ 1)
        device, dtype: torch settings

    Returns:
        povms: [Π₀, Π₁], each (hilbert_dim, hilbert_dim), and diagonal.
    """

    n = th.arange(hilbert_dim, dtype=dtype)
    diag_no_click = (1 - eta) ** n
    diag_click = 1.0 - diag_no_click

    Pi0 = th.diag(diag_no_click)
    Pi1 = th.diag(diag_click)

    return [Pi0, Pi1]


def povm_fidelity(povm_a: Tensor, povm_b: Tensor) -> float:
    """
    Computes the fidelity of two POVM elements according to the formula in 
    Eq. 13 of https://doi.org/10.1088/2058-9565/ad8511
    and Eq. 40 of doi:10.1088/1367-2630/14/11/115005.

    Args:
        povm_a: Tensor of shape (M, M) where M is the Hilbert space dimension
        povm_b: Tensor of shape (M, M) where M is the Hilbert space dimension
    
    Returns:
        Fidelity between the two POVM elements
    """
    F = fidelity(Qobj(povm_a), Qobj(povm_b))**2 
    F /= (trace(povm_a)*trace(povm_b))
    return F.real.item()


def diag_povm_fidelity(povm_a: Tensor, povm_b: Tensor) -> float:
    """
    A version of the povm_fidelity function that exploits the fact that 
    diagonal POVMs (matrices) don't require the full matrix for computations i.e.
    sqrt of a diagonal matrix is computed by just taking the sqrt of diagonal elements. 

    povm_a: An (M,1) tensor that is the diagonal elements of the POVM element
    povm_b: An (M,1) tensor that is the diagonal elements of the POVM element

    Returns:
        Fidelity between the two POVM elements
    """
    sqrt_a = th.sqrt(povm_a)
    F = th.sqrt( sqrt_a * povm_b * sqrt_a )
    F = th.sum(F)**2
    F /= ( th.sum(povm_a) * th.sum(povm_b) )

    return F.real.item()