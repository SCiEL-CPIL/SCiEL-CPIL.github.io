# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration: damped Duffing form with mild nonlinearity.
    Uses physically-motivated terms scaled for good BFGS conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Scale to unit range for conditioning
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    X = x / 0.55
    V = v / 0.25
    
    # Physically motivated Duffing oscillator terms:
    # a = -(k1*X + k3*X^3) - (c1*V + c3*V^3) + coupling + offset
    # Keep terms well-scaled and limited to avoid ill-conditioning
    
    # Linear restoring and damping (dominant)
    t1 = X
    t2 = V
    
    # Cubic nonlinearity (Duffing)
    t3 = X**3
    t4 = V**3
    
    # Cross coupling (weak)
    t5 = X * V
    
    # Quadratic corrections (weak)
    t6 = X**2
    t7 = V**2
    
    # Higher-order cross terms (very weak)
    t8 = X**2 * V
    t9 = V**2 * X
    
    # Constant
    t10 = np.ones_like(X)
    
    a = (p[0] * t1 +
         p[1] * t2 +
         p[2] * t3 +
         p[3] * t4 +
         p[4] * t5 +
         p[5] * t6 +
         p[6] * t7 +
         p[7] * t8 +
         p[8] * t9 +
         p[9] * t10)
         
    return a
# EVOLVE-BLOCK-END
