# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a whitened Fourier basis.
    The basis consists of 1 constant term and 5 pairs of sine/cosine terms
    with frequencies scaled to the domain [-3, 3].
    Columns are normalized to unit L2 norm to ensure a well-conditioned
    Hessian for the BFGS optimizer.
    """
    _T0 = time.time()
    _DEADLINE = 140.0

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([])

    # Scale x to [-pi, pi] for standard Fourier frequencies
    # x in [-3, 3] -> u in [-pi, pi]
    u = x * (np.pi / 3.0)
    
    # Construct basis matrix V (n x 10)
    # Col 0: 1
    # Col 1: cos(1*u), Col 2: sin(1*u)
    # Col 3: cos(2*u), Col 4: sin(2*u)
    # ...
    # Col 9: sin(5*u)
    
    V = np.zeros((n, 10), dtype=np.float64)
    V[:, 0] = 1.0
    
    for k in range(1, 6):
        V[:, 2*k - 1] = np.cos(k * u)
        V[:, 2*k] = np.sin(k * u)
        
    # Whitening: normalize each column to unit norm
    # This makes the design matrix orthonormal (approximately, if n is large enough
    # and samples are uniform, the empirical Gram matrix V.T @ V / n is close to I)
    # Normalizing columns ensures the Hessian 2 * V.T @ V is close to 2 * I * (norms^2)
    # Actually, if we normalize columns to unit L2 norm on the *training* set,
    # V.T @ V = I exactly.
    
    norms = np.linalg.norm(V, axis=0)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    V_whitened = V / norms
    
    y = V_whitened @ params

    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)

    if time.time() - _T0 > _DEADLINE:
        return y

    return y
# EVOLVE-BLOCK-END
