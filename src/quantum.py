from itertools import product
from math import exp, comb

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


def pnr_povm(hilbert_dim: int, N: int, eta: float, dtype=th.float32) -> list[th.Tensor]:
    """
    Returns the POVM of a PNR detector (up to N detected photons)
    with quantum efficiency `eta` and a Fock-space truncation of `hilbert_dim`.

    Based on formula (See Millers thesis section 6.1 `Detector POVMs` for more.):
        ⟨n|Π_k|n⟩ = C(n, k) * η^k * (1 - η)^(n - k)
    for 0 <= k <= min(n, N).

    Args:
        hilbert_dim: Hilbert space cutoff (max photon number considered).
        N: maximum number of detected photons (PNR resolution).
        eta: detection efficiency (0 ≤ eta ≤ 1).

    Returns:
        povms: list of N+1 tensors [Π₀, Π₁, ..., Π_N], each (hilbert_dim, hilbert_dim), diagonal.
    """
    povm = [th.zeros((hilbert_dim, hilbert_dim), dtype=dtype) for _ in range(N + 1)]

    # Outer loop over input photon number n
    for n in range(hilbert_dim):
        # Inner loop over detected photon number k
        for k in range(min(n, N) + 1):
            p = comb(n, k) * (eta ** k) * ((1 - eta) ** (n - k))
            povm[k][n, n] = p  # ⟨n|Π_k|n⟩ = P(k|n)

    # Merge higher counts into Π_N (saturation) if Hilbert space is larger than detector resolution
    # I - ΣE_i for 0 ≤ eta ≤ N-1
    if hilbert_dim > N:
        for n in range(N + 1, hilbert_dim):
            p = 1 - sum(comb(n, k) * (eta ** k) * ((1 - eta) ** (n - k)) for k in range(N))
            povm[N][n, n] = p

    return povm 
    

def photodetector_povm(hilbert_dim: int, eta: float, dtype=th.float32) -> list[th.Tensor]:
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

def rand_DV_diag_povm(num_qubits: int, dtype=th.float32) -> list[th.Tensor]:
    """
    Returns a POVM with 2^num_qubits all diagonal elements with Hilbert space 
    dimension of size `2^num_qubits.

    Args:
        num_qubits: Number of qubits (two level systems)

    Returns:
        povms: list of N+1 tensors [Π₀, Π₁, ..., Π_N], each (hilbert_dim, hilbert_dim), diagonal.
    """
    num_povm_elements = 2**num_qubits
    hilbert_dim = 2**num_qubits
    povm = []
    
    for _ in range(num_povm_elements):
        p = th.rand(hilbert_dim, dtype=dtype)  # create random 1D vector
        p = p / p.sum()  # normalize to probability vector
        p = th.diag(p)   # (2^N, 2^N) diagonal matrix
        povm.append(p)    

    return povm 


def get_qubit_probe_states(num_qubits: int, return_dm: bool = False, povm_is_diag: bool = True, device: str = 'cpu'):
    """
    Generate all probe states for `num_qubits` using the standard basis |0>, |1>, |+>, |+i>.

    Args:
        num_qubits (int): Number of qubits.
        return_dm (bool): If True, return density matrices instead of kets.
        povm_is_diag (bool): Wether not the detector POVM is assumed to be diagonal in the computational basis. 
        device (str): Torch device.

    Returns:
        torch.Tensor: Tensor of shape (4**num_qubits, 2**num_qubits) for kets,
                      or (4**num_qubits, 2**num_qubits, 2**num_qubits) for density matrices.
    """
    # single-qubit states
    zero = th.tensor([1.0, 0.0], dtype=th.complex64, device=device)
    one = th.tensor([0.0, 1.0], dtype=th.complex64, device=device)
    plus = (zero + one) / th.sqrt(th.tensor(2.0))
    plus_i = (zero + 1j * one) / th.sqrt(th.tensor(2.0))
    
    if povm_is_diag:
        basis = [zero, one]
    else:
        basis = [zero, one, plus, plus_i]
    
    # generate all combinations of basis states for num_qubits
    states = []
    for combo in product(basis, repeat=num_qubits):
        ket = combo[0]
        for b in combo[1:]:
            ket = th.kron(ket, b)
        if return_dm:
            rho = ket.unsqueeze(-1) @ ket.conj().unsqueeze(0)
            states.append(rho)
        else:
            states.append(ket)
    
    return th.stack(states)

def apply_t1_to_populations(probes: th.Tensor, gamma: float, num_qubits: int) -> th.Tensor:
    """
    Apply T1 relaxation to diagonal populations of D probe states.

    probes: (D, M) tensor, M = 2**N
    gamma: scalar T1 probability

    Returns:
        P_out: (D, M)
    """
    assert 0 <= gamma <= 1, "Relaxation coefficient must be between 0 and 1."
    D, M = probes.shape
    N = num_qubits

    # Reshape to (D, 2, 2, ..., 2)
    probs = probes.view(D, *([2] * N))

    for _ in range(N):
        # apply T1 along one qubit axis at a time
        p0 = probs.select(-1, 0)
        p1 = probs.select(-1, 1)

        new_p0 = p0 + gamma * p1
        new_p1 = (1 - gamma) * p1

        probs = th.stack([new_p0, new_p1], dim=-1)

    return probs.view(D, M)


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

def compute_expectation_value(state: th.Tensor, operator: th.Tensor) -> float:
    """
    Compute the expectation value of an operator given a quantum state.
    
    Args:
        state (Tensor): Either a ket |psi> (shape [dim]) or a density matrix rho (shape [dim, dim])
        operator (Tensor): Hermitian operator E (shape [dim, dim])
    
    Returns:
        float: expectation value <E>
    """
    assert operator.dim() == 2, "Operator must have 2 dimensions."
    
    if state.dim() == 1:  # ket
        # <psi|E|psi>
        exp_val = th.conj(state) @ operator @ state
    elif state.dim() == 2:  # density matrix
        # Tr[rho*E]
        exp_val = th.trace(state @ operator)
    else:
        raise ValueError(f"State must be 1D (ket) or 2D (density matrix), got shape {state.shape}")
    
    return exp_val.real.item()