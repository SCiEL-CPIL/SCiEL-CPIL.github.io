"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Sum of Fixed Gaussians basis.
    
    Structural Change: Instead of global polynomial/trig basis, we use local
    Gaussian kernels. This allows the model to fit non-periodic, smooth,
    potentially multi-modal functions more effectively without the ill-conditioning
    issues of high-degree polynomials.
    
    Model:
    f(x) = sum_{i=0}^{9} p_i * exp(-0.5 * ((x - c_i) / sigma_i)^2)
    
    Centers c_i are fixed, uniformly spaced in [0.5, 5.5].
    Widths sigma_i are fixed to 0.6 to ensure overlap.
    Parameters p_i are the amplitudes.
    
    Since the model is linear in p_i, BFGS will converge to the optimal
    least-squares solution for this basis.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback to simple linear if running out of time
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Fixed Gaussian centers covering the domain [0.1, 6]
    # 10 centers, spaced by ~0.55
    centers = np.linspace(0.5, 5.5, 10)
    # Fixed standard deviations
    sigmas = np.full(10, 0.6)
    
    # Compute basis functions
    # x_arr shape: (N,)
    # centers shape: (10,)
    # We need to compute (x_i - c_j) for all i, j
    # Using broadcasting: x_arr[:, np.newaxis] - centers[np.newaxis, :]
    diffs = x_arr[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian values: exp(-0.5 * (diffs / sigma)^2)
    # sigmas shape (10,) -> broadcast to (1, 10)
    gaussians = np.exp(-0.5 * (diffs / sigmas[np.newaxis, :]) ** 2)
    
    # Linear combination: result = gaussians @ p
    # gaussians shape: (N, 10)
    # p shape: (10,)
    result = gaussians @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to simple linear if numerical issues arise
        result = p[0] * x_arr + p[1]
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
