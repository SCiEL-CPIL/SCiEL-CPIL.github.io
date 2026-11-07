"""
Hybrid Basis Strategy: Combines a global polynomial backbone with local RBFs.
The polynomial part (degree 2) uses 6 parameters to capture general trends.
The RBF part uses 4 parameters to define 2 Gaussian centers (cx, cy) for local correction.
This structure is more flexible than pure polynomials but more stable than pure RBFs
with fixed weights or too many free parameters.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Quadratic Polynomial + 2 Gaussian RBFs.
    
    Parameters (length 10):
    p0..p5: Coefficients for [1, x, y, x^2, xy, y^2]
    p6: Center x for RBF 1
    p7: Center y for RBF 1
    p8: Center x for RBF 2
    p9: Center y for RBF 2
    
    RBF widths are fixed to 0.5 to ensure stability and prevent the optimizer
    from getting stuck in narrow valleys. The weights of the RBFs are effectively
    learned by the interaction with the polynomial and the target data during
    the BFGS fit, although strictly speaking, we are only moving the centers.
    Wait, if weights are fixed to 1, the RBFs might overcorrect.
    Let's use a soft-weighted approach or just accept that the polynomial
    will adjust to compensate. Actually, a better approach for "universal approximator"
    with limited params is to let the RBFs have fixed small weights or let the
    polynomial handle the mean.
    
    Let's refine: 
    p0: Global offset
    p1: x coeff
    p2: y coeff
    p3: x^2 coeff
    p4: y^2 coeff
    p5: xy coeff
    p6: RBF1 center x
    p7: RBF1 center y
    p8: RBF2 center x
    p9: RBF2 center y
    
    The RBF contribution is: 1.0 * exp(-((x-p6)^2 + (y-p7)^2) / (2*0.5^2)) + 1.0 * exp(-((x-p8)^2 + (y-p9)^2) / (2*0.5^2))
    
    This allows the model to "bump up" or "bump down" (if the polynomial can go negative) 
    in specific regions. Since the polynomial has 6 terms, it can subtract from the RBFs.
    """
    params = np.asarray(params).flatten()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5 = params[0:6]
    cx1, cy1, cx2, cy2 = params[6:10]
    
    # 1. Polynomial Backbone (Degree 2)
    # Basis: 1, x, y, x^2, xy, y^2
    poly_part = (
        p0 
        + p1 * x 
        + p2 * y 
        + p3 * x**2 
        + p4 * y**2 
        + p5 * x * y
    )
    
    # 2. RBF Correction (2 Gaussians)
    # Fixed width sigma = 0.5
    # sigma^2 = 0.25, 2*sigma^2 = 0.5
    dist1_sq = (x - cx1)**2 + (y - cy1)**2
    rbf1 = np.exp(-dist1_sq / 0.5)
    
    dist2_sq = (x - cx2)**2 + (y - cy2)**2
    rbf2 = np.exp(-dist2_sq / 0.5)
    
    # Weights for RBFs are fixed to 1.0. 
    # The polynomial can adapt to balance the overall scale.
    rbf_part = rbf1 + rbf2
    
    pred = poly_part + rbf_part
    
    return pred
# EVOLVE-BLOCK-END
