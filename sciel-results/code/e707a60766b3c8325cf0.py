# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a scaled polynomial basis.
    Scaling inputs to unit magnitude improves the conditioning of the
    optimization problem, allowing the fitter to better distinguish
    between linear and nonlinear contributions.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Scale inputs to approximately [-1, 1] range to improve conditioning
    # Training ranges: x ~ [-0.57, 0.53], v ~ [-0.18, 0.28]
    x_scale = 0.6
    v_scale = 0.3
    
    X = x / x_scale
    V = v / v_scale
    
    # Define basis functions
    # Linear terms
    t1 = X
    t2 = V
    
    # Quadratic terms
    t3 = X**2
    t4 = V**2
    t5 = X * V
    
    # Cubic terms (Duffing-like and nonlinear damping)
    t6 = X**3
    t7 = V**3
    t8 = X**2 * V
    t9 = X * V**2
    
    # Constant offset
    t10 = np.ones_like(X)
    
    # Linear combination
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
