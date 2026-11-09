"""RBF + polynomial hybrid for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF + polynomial hybrid model for nonlinear oscillator acceleration.
    
    Combines local RBF basis functions in (x, v) phase space with global
    polynomial terms and sinusoidal drive terms. The RBF terms capture
    local nonlinear coupling that a global polynomial might miss.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    n = len(t)
    
    # Normalize x and v to roughly [-1, 1]
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    xn = (x - x_center) / x_half
    vn = (v - v_center) / v_half
    
    # Clip for safety
    xn = np.clip(xn, -2.0, 2.0)
    vn = np.clip(vn, -2.0, 2.0)
    
    # RBF centers in (xn, vn) space - 4 local basis functions
    # Placed at corners and center of the phase space
    rbf_centers = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [-0.5,  0.5],
        [ 0.5,  0.5],
    ])
    
    # RBF width (inverse bandwidth)
    sigma = 0.8
    
    # Compute RBF values: 4 RBF terms
    rbf_vals = np.zeros((n, 4), dtype=np.float64)
    for i, (cx, cv) in enumerate(rbf_centers):
        dx = xn - cx
        dv = vn - cv
        r2 = dx**2 + dv**2
        rbf_vals[:, i] = np.exp(-r2 / (2.0 * sigma**2))
    
    # Polynomial terms in x: 1, x, x^2 (3 terms)
    poly_x = np.column_stack((
        np.ones_like(xn),
        xn,
        xn**2
    ))
    
    # Polynomial term in v: v (1 term)
    poly_v = np.column_stack((vn,))
    
    # Drive terms: sin(t), cos(t) (2 terms)
    drive = np.column_stack((
        np.sin(t),
        np.cos(t)
    ))
    
    # Total basis: 4 RBF + 3 poly_x + 1 poly_v + 2 drive = 10 terms
    basis = np.column_stack((
        rbf_vals,        # 4 RBF terms
        poly_x,          # 3 polynomial x terms
        poly_v,          # 1 polynomial v term
        drive            # 2 drive terms
    ))
    
    # Linear combination
    result = basis @ p
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
