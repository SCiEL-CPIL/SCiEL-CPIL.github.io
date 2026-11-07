"""
Symbolic regression using a hybrid Chebyshev + RBF basis.

Design:
- 5 Chebyshev polynomials T_0..T_4 on scaled domain [-1, 1] for smooth global structure.
- 5 Radial Basis Functions (Gaussians) with centers spanning [-2, 2] for local features.
- This hybrid basis can capture both polynomial-like trends and localized bumps/peaks.
- BFGS fits the 10 params from all-ones start.
- Includes a deadline guard to stay within the 150s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (uniformly from [-2, 2])
    params: array of 10 coefficients
    Returns: predicted values, same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a simple constant prediction as fallback
        return np.full_like(x, params[0])
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    b4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Radial Basis Functions (Gaussians)
    # Centers evenly spaced across [-2, 2] to cover the full input range
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    sigma = 0.7  # Width parameter, tuned for domain [-2, 2]
    
    # Compute Gaussians: shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma)**2)
    
    # Build full basis matrix of shape (N, 10)
    # Columns 0-4: Chebyshev
    # Columns 5-9: Gaussians
    B = np.column_stack((b0, b1, b2, b3, b4, gaussians))
    
    # Compute prediction
    pred = B @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
