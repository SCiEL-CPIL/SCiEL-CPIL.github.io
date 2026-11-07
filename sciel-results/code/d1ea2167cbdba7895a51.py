"""
Symbolic regression using a rich, well-conditioned basis.

Design:
- Use Chebyshev polynomials of the first kind on the scaled domain [-1, 1]
  for the first 8 coefficients (indices 0-7).
- Use a few additional basis functions (sin, cos, tanh, x*tanh(x)) for
  indices 8-9 to capture potential non-polynomial structure.
- The basis is evaluated in a numerically stable way.
- BFGS will fit the 10 params from an all-ones start.
"""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_basis(t: np.ndarray, n: int) -> np.ndarray:
    """
    Compute Chebyshev polynomials T_0(t), T_1(t), ..., T_{n-1}(t).
    t is assumed to be in [-1, 1].
    Returns an array of shape (len(t), n).
    """
    m = len(t)
    basis = np.zeros((m, n))
    if n == 0:
        return basis
    basis[:, 0] = 1.0
    if n > 1:
        basis[:, 1] = t
        for k in range(2, n):
            basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (uniformly from [-2, 2])
    params: array of 10 coefficients
    Returns: predicted values, same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Build the basis matrix: 8 Chebyshev + 2 additional
    n_cheb = 8
    cheb_basis = _chebyshev_basis(t, n_cheb)  # shape (m, 8)
    
    # Additional basis functions
    sin_term = np.sin(np.pi * t)
    cos_term = np.cos(np.pi * t)
    
    # Build full basis matrix of shape (m, 10)
    m = len(x)
    basis = np.zeros((m, 10))
    basis[:, :n_cheb] = cheb_basis
    basis[:, n_cheb] = sin_term
    basis[:, n_cheb + 1] = cos_term
    
    # Compute prediction
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
