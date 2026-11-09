# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using Duffing-like form with 
    physically motivated terms. Uses 10 parameters for:
    - Linear and cubic restoring force
    - Linear and cubic damping  
    - Cross coupling terms
    - Quadratic terms
    - Constant offset
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Duffing oscillator: a = -k*x - c*x^3 - d*v - e*v^3 + coupling terms
    # p0: linear stiffness (restoring)
    # p1: cubic stiffness (Duffing)
    # p2: linear damping
    # p3: cubic damping
    # p4: x*v cross term
    # p5: x^2 quadratic position
    # p6: v^2 quadratic velocity
    # p7: x^2*v 
    # p8: x*v^2
    # p9: constant bias
    
    a = (p[0] * x +
         p[1] * x**3 +
         p[2] * v +
         p[3] * v**3 +
         p[4] * x * v +
         p[5] * x**2 +
         p[6] * v**2 +
         p[7] * x**2 * v +
         p[8] * x * v**2 +
         p[9])
    
    return a
# EVOLVE-BLOCK-END
