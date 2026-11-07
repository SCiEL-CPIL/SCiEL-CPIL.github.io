import numpy as np
import time

# Hard deadline guard: return safely before the 150 s limit.
_START = time.time()
_DEADLINE = _START + 140.0

# Precompute RBF centers and widths for the domain [0.1, 6]
# Use 10 evenly-spaced centers covering the domain
_X_MIN = 0.1
_X_MAX = 6.0
_N_CENTERS = 10
# Centers: evenly spaced from slightly inside the domain
_CENTERS = np.linspace(_X_MIN + 0.2, _X_MAX - 0.2, _N_CENTERS)
# Width: chosen so adjacent RBFs have significant overlap
# Domain width = 5.9, with 10 centers, spacing ~ 0.656
# Use width = 0.45 for good local coverage with overlap
_WIDTH = 0.45
_INV_2W2 = 1.0 / (2.0 * _WIDTH * _WIDTH)

def _rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build a Gaussian RBF basis with 10 fixed centers and fixed width.
    Each basis function is phi_i(x) = exp(-(x - c_i)^2 / (2 * width^2)).
    This gives local, well-conditioned basis functions where each parameter
    acts as an independent local weight.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    N = x.size
    if N == 0:
        return np.empty((0, _N_CENTERS), dtype=np.float64)
    
    # Compute (x - c_i)^2 for all i
    # x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    sq = diff * diff
    
    # phi_i(x) = exp(-sq / (2 * width^2))
    B = np.exp(-sq * _INV_2W2)
    
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form predictor using Gaussian RBF basis.
    Linear in parameters, so BFGS converges quickly to the global minimum.
    RBFs provide local, well-conditioned basis functions that can capture
    non-smooth features and sharp variations better than global polynomials.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != _N_CENTERS:
        return np.zeros_like(x)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(x)
    
    B = _rbf_basis(x)
    
    out = B @ p
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x)
    return out
