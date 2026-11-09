"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()

# Fixed RBF centers in the native x-domain [-3, 3].
# 6 centers spaced by 1.0 unit.
_RBF_CENTERS = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
_RBF_WIDTH = 0.6  # Gaussian width (sigma)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Low-order Chebyshev polynomials (global trend) + 
    Gaussian RBFs (local features).
    
    Total parameters: 10
    params[0..3] : 4 Chebyshev coefficients T_0..T_3
    params[4..9] : 6 Gaussian RBF amplitudes at fixed centers
    
    This design ensures that all 10 parameters are utilized, combining 
    global smoothness with local flexibility.
    """
    # Deadline guard
    if time.time() - _T0 > 145.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
    
    # Ensure params is at least length 10, pad with zeros if necessary
    # (Though evaluator should provide 10)
    p = np.zeros(10, dtype=np.float64)
    if len(params) >= 10:
        p[:] = params[:10]
    else:
        p[:len(params)] = params

    # ---- Chebyshev Part (Global Trend) ----
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t = np.clip(t, -1.0, 1.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)

    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 2*t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 2.0 * t

    # Linear combination of global terms
    result = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3

    # ---- Gaussian RBF Part (Local Detail) ----
    # Compute RBF matrix: 6 columns, one per center
    # phi_i(x) = exp(-((x - c_i) / sigma)^2)
    
    # Vectorized computation
    # diff shape: (N, 6)
    diff = x[:, None] - _RBF_CENTERS[None, :]
    r2 = diff * diff
    rbf_matrix = np.exp(-r2 / (_RBF_WIDTH * _RBF_WIDTH))
    
    # rbf_matrix shape: (N, 6)
    # amps shape: (6,)
    amps = p[4:10]
    
    rbf_contrib = rbf_matrix @ amps
    
    result = result + rbf_contrib

    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    return result
# EVOLVE-BLOCK-END
