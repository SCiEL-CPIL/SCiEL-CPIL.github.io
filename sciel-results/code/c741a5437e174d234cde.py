"""
Symbolic regression with a hybrid Chebyshev + RBF basis.
The hidden function is approximated by a linear combination of:
- 5 Chebyshev polynomials (smooth global structure)
- 5 Radial Basis Functions (local flexibility)
This gives BFGS a richer function space to fit diverse smooth targets.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track elapsed time to avoid timeout
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds total budget
_SAFETY_MARGIN = 5.0  # seconds to spare

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values (array with one prediction per point)
    """
    # Safety check: if we're close to deadline, just return a simple prediction
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: linear fit as a safe fallback
        return params[0] + params[1] * x
    
    n = len(x)
    n_params = 10
    
    # --- Chebyshev basis (first 5 params) ---
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # T_0 = 1, T_1 = t, T_k = 2*t*T_{k-1} - T_{k-2}
    T0 = np.ones(n)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # --- RBF basis (last 5 params) ---
    # Centers spread across [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # Width parameter: moderate spread to cover local features
    width = 1.2
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * width^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    RBF = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Combine: [T0, T1, T2, T3, T4, RBF0, RBF1, RBF2, RBF3, RBF4]
    basis = np.column_stack([
        T0, T1, T2, T3, T4,
        RBF[:, 0], RBF[:, 1], RBF[:, 2], RBF[:, 3], RBF[:, 4]
    ])
    
    # Linear combination
    result = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
