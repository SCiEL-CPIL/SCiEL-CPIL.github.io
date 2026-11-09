"""
Strategy: Complete Separable Polynomial Basis (Degree 3).
Instead of local Radial Basis Functions, this strategy uses a global polynomial basis 
that spans a 10-dimensional subspace of functions. The basis functions are:
1, x, y, x^2, y^2, xy, x^3, x^2y, y^2x, y^3.

This approach is linear in the parameters, ensuring that BFGS (or any linear solver) 
can find the global minimum for the fixed basis. Unlike the RBF approach which relies 
on local coverage, the polynomial basis captures global trends, interactions, and 
curvature directly. For smooth functions on [-1, 1]^2, a degree-3 polynomial often 
provides a strong approximation, and using all 10 parameters for coefficients maximizes 
the expressiveness within the parameter budget.

The key difference from the current RBF strategy is the "global vs local" nature of the 
basis. Polynomials are global (every term affects the whole domain), while RBFs are local. 
This can be advantageous if the target function has strong global structure (e.g., 
interactions or trends) that local bumps struggle to represent without many centers.
"""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure we have at least 10 parameters
    # The contract states params is length 10, but we slice to be safe
    if len(params) < 10:
        # Pad with zeros if somehow short (should not happen)
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = p
    
    # Compute basis terms
    # To optimize speed, we compute powers once
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    y2x = y2 * x
    
    # Construct prediction:
    # Index 0: 1 (bias)
    # Index 1: x
    # Index 2: y
    # Index 3: x^2
    # Index 4: y^2
    # Index 5: xy
    # Index 6: x^3
    # Index 7: x^2y
    # Index 8: y^2x
    # Index 9: y^3
    
    pred = w0 + w1 * x + w2 * y + w3 * x2 + w4 * y2 + w5 * xy + w6 * x3 + w7 * x2y + w8 * y2x + w9 * y3
    
    # Safety check for time
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong (unlikely to be needed)
         return p[0] * x + p[1] * y + p[2]
         
    return pred
