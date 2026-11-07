import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product-based feature expansion:
    10 terms, each = Chebyshev polynomial T_k(t) * activation_k(t)
    
    This creates a rich set of nonlinear basis functions that are:
    - Globally stable (Chebyshev polynomials are bounded on [-1,1])
    - Nonlinear (activated by tanh, sin, or polynomial factors)
    - Orthogonal-like (Chebyshev basis provides good conditioning)
    
    Basis functions:
    phi_0 = T0 * 1 = 1
    phi_1 = T1 * tanh(t)
    phi_2 = T2 * t
    phi_3 = T3 * tanh(t)
    phi_4 = T4 * t^2
    phi_5 = T5 * sin(t)
    phi_6 = T6 * tanh(t)
    phi_7 = T7 * t
    phi_8 = T8 * sin(t)
    phi_9 = T9 * tanh(t)
    
    where t = x/3 maps x from [-3,3] to [-1,1]
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T0..T9 via recurrence
    T = np.zeros((10, len(t)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Define activations for each term
    # Mix of tanh (smooth sigmoid-like), sin (periodic), and polynomial factors
    tanh_t = np.tanh(t)
    sin_t = np.sin(t)
    t_sq = t * t
    t_cu = t_sq * t
    
    # Build the 10 product terms
    phi = np.zeros((10, len(t)))
    phi[0, :] = T[0, :] * 1.0           # constant
    phi[1, :] = T[1, :] * tanh_t        # linear * tanh
    phi[2, :] = T[2, :] * t             # quadratic * linear
    phi[3, :] = T[3, :] * tanh_t        # cubic * tanh
    phi[4, :] = T[4, :] * t_sq          # quartic * quadratic
    phi[5, :] = T[5, :] * sin_t         # quintic * sin
    phi[6, :] = T[6, :] * tanh_t        # sextic * tanh
    phi[7, :] = T[7, :] * t             # septic * linear
    phi[8, :] = T[8, :] * sin_t         # octic * sin
    phi[9, :] = T[9, :] * tanh_t        # nonic * tanh
    
    # Linear combination
    y = params @ phi
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
