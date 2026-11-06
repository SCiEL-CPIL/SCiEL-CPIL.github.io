"""
Strategy: Exploit the physical antisymmetry of the unresolved fast-scale forcing.
In the two-scale Lorenz-96 model, the fast subsystem's response to the slow state X
is typically antisymmetric (odd function) due to the symmetry of the equations.
Specifically, if X -> -X, the forcing -> -Forcing.
Therefore, the polynomial approximation should only contain odd powers: x, x^3, x^5.
This reduces the number of free parameters from 6 to 3, improving the bias-variance tradeoff
and generalization to the unseen F=10 regime.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses an odd-power-only sparse model: y = c1*x + c3*x^3 + c5*x^5.
    Returns 6 coefficients [c0, c1, c2, c3, c4, c5] where even powers are 0.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Basic sanity checks
    if len(x) == 0:
        return np.zeros(6)
    
    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return np.zeros(6)
    
    # We want to fit: y = c1*x + c3*x^3 + c5*x^5
    # This is a linear least squares problem for c1, c3, c5.
    # Design matrix V has columns: [x, x^3, x^5]
    
    # Check time budget
    if time.time() - start_time > deadline:
        return np.zeros(6)
    
    try:
        # Construct the sparse Vandermonde matrix for odd powers
        # Columns: x^1, x^3, x^5
        V = np.column_stack((x, x**3, x**5))
        
        # Solve least squares
        # Using lstsq for robustness
        coeffs, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
        
        # coeffs are [c1, c3, c5]
        
        # Verify finite
        if not np.all(np.isfinite(coeffs)):
            return np.zeros(6)
            
        # Map to full degree-5 coefficient vector [c0, c1, c2, c3, c4, c5]
        full_coeffs = np.zeros(6)
        full_coeffs[1] = coeffs[0]
        full_coeffs[3] = coeffs[1]
        full_coeffs[5] = coeffs[2]
        
        return full_coeffs
        
    except Exception:
        # Fallback: simple linear fit if sparse fit fails
        try:
            c1 = np.dot(x, y) / (np.dot(x, x) + 1e-12)
            return np.array([0.0, c1, 0.0, 0.0, 0.0, 0.0])
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
