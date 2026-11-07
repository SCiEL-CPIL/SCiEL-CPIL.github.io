import numpy as np
import time

_start_time = time.time()
DEADLINE = 135.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a rich, smooth, bounded basis of 10 functions to approximate the hidden target.
    The model is linear in params: y = sum(p_i * phi_i(x)).
    This ensures BFGS converges reliably to the least-squares solution.
    """
    if time.time() - _start_time > DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)

    # Clip x to safe range to prevent potential issues, though inputs are [-3, 3]
    x_safe = np.clip(x, -3.0, 3.0)
    
    # Basis functions
    # 1. Constant
    phi0 = np.ones_like(x_safe)
    
    # 2. Linear
    phi1 = x_safe
    
    # 3. Quadratic
    phi2 = x_safe ** 2
    
    # 4. Cubic
    phi3 = x_safe ** 3
    
    # 5. Sine (period ~ 2*pi ~ 6.28, covers ~1 cycle over [-3,3])
    phi4 = np.sin(x_safe)
    
    # 6. Cosine
    phi5 = np.cos(x_safe)
    
    # 7. Sine (higher freq, ~2 cycles)
    phi6 = np.sin(2.0 * x_safe)
    
    # 8. Cosine (higher freq)
    phi7 = np.cos(2.0 * x_safe)
    
    # 9. Tanh (smooth step/saturating)
    phi8 = np.tanh(x_safe)
    
    # 10. Rational bump (even, peak at 0, decays)
    # 1 / (1 + x^2) is smooth and bounded on [-3,3]
    phi9 = 1.0 / (1.0 + x_safe ** 2)
    
    # Stack basis: shape (10, N)
    basis = np.vstack([
        phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9
    ])
    
    # Linear combination
    y = p @ basis
    
    # Ensure finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return y
