from math import gamma, sqrt, exp
from torch import Tensor, tensor, linalg, dtype, complex128

def coherent_ket(
    alpha: complex, dim: int, dtype: dtype = complex128
) -> Tensor:
    """
    Returns the ket of the coherent 
    state (displaced vacuum state): D(α)|0⟩
    Note that math.gamma(n) computes (n-1)! not n! so be careful.
    """

    const = exp(-(abs(alpha) ** 2) / 2.0)

    ket = tensor(
        [const * (alpha ** (n)) / sqrt(gamma(n + 1)) for n in range(dim)], dtype=dtype
    )

    norm = linalg.norm(ket)
    ket /= norm  # normalize ket

    return ket