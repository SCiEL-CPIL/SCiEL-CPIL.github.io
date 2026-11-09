"""
Symbolic regression using a hybrid basis: low-order polynomial + sigmoid.
This structural change moves from a pure polynomial representation to one that
can model step-like or saturating behaviors, providing a different functional
geometry that may better fit the hidden target function.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = p0 + p1*x + p2*x^2 + p3 * sigmoid(p4*(x - p5))
    
    Parameters:
        p0: constant offset
        p1: linear coefficient
        p2: quadratic coefficient
        p3: sigmoid amplitude
        p4: sigmoid steepness
        p5: sigmoid center
    
    This hybrid representation combines a low-order polynomial (for smooth
    trends) with a single learnable sigmoid (for non-linear transitions or
    saturation), offering a fundamentally different functional geometry than
    pure polynomial bases.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p is 1D
    if p.ndim != 1:
        p = p.flatten()
    
    # Extract parameters
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    
    # Compute polynomial part
    poly_part = p0 + p1 * x + p2 * x**2
    
    # Compute sigmoid part with numerical stability
    z = p4 * (x - p5)
    # Use stable sigmoid: sigmoid(z) = 1 / (1 + exp(-z))
    # For large positive z, sigmoid ~ 1; for large negative z, sigmoid ~ 0
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -500, 500)
    sigmoid_part = 1.0 / (1.0 + np.exp(-z_clipped))
    
    # Combine
    return poly_part + p3 * sigmoid_part
# EVOLVE-BLOCK-END
