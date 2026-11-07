"""
Strategy: Hybrid Global-Local Adaptive Basis.
Combines a global polynomial/trig basis (for smooth, global trends) with local
Gaussian RBFs (for localized features). The 10 parameters are split:
- p[0:4]: coefficients for global basis (1, x, x^2, sin(x))
- p[4:7]: centers for 3 local Gaussian RBFs
- p[7:10]: widths for 3 local Gaussian RBFs
The coefficients for the local RBFs are fixed to 1.0 to avoid scaling issues.
This allows BFGS to fit both global structure and local deviations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.asarray(x).ravel()
    
    # Global basis: constant, linear, quadratic, sin
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = np.sin(x)
    
    # Local basis: 3 Gaussian RBFs
    # Centers: p[4], p[5], p[6]
    # Widths:   p[7], p[8], p[9]
    c1 = params[4]
    c2 = params[5]
    c3 = params[6]
    w1 = params[7]
    w2 = params[8]
    w3 = params[9]
    
    # Ensure widths are positive and not too small to avoid numerical issues
    # Use a soft lower bound on width to prevent division by zero or overflow
    w1 = np.maximum(w1, 0.1)
    w2 = np.maximum(w2, 0.1)
    w3 = np.maximum(w3, 0.1)
    
    g1 = np.exp(-((x - c1) ** 2) / (2 * w1 ** 2))
    g2 = np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))
    g3 = np.exp(-((x - c3) ** 2) / (2 * w3 ** 2))
    
    # Combine: global terms with coefficients, local terms with fixed coeff 1.0
    # p[0]*b0 + p[1]*b1 + p[2]*b2 + p[3]*b3 + 1.0*g1 + 1.0*g2 + 1.0*g3
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            g1 + g2 + g3)
# EVOLVE-BLOCK-END
