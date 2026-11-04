import numpy as np
import scipy.optimize as opt
import time

# EVOLVE-BLOCK-START

# Pre-compute Gauss-Laguerre nodes and weights
# We need to integrate polynomials of degree up to 49 (x * P^2 where P is deg 24).
# Gauss-Laguerre with N points integrates exactly polynomials up to degree 2N-1.
# 2N - 1 >= 49 => N >= 25.
# Use N=60 for high accuracy.
GL_N = 60
x_gl, w_gl = np.polynomial.laguerre.laggauss(GL_N)

def _compute_ratios(roots):
    """
    Compute I0 = int e^-x P^2 and I1 = int e^-x x P^2
    Returns (I1/I0, I0, I1)
    """
    # roots: array of shape (12,)
    # x_gl: array of shape (60,)
    
    # Calculate P(x) = prod (x - r_i)^2
    # diff[i, j] = x_gl[j] - roots[i]
    diff = x_gl[None, :] - roots[:, None]
    
    # P(x) = prod_i (x - r_i)^2
    # To improve numerical stability, we can sum logs
    # But direct product is likely fine for these magnitudes.
    # Let's check magnitude.
    # Max term ~ (100)^2 = 10000.
    # Product of 12 terms ~ 10000^12 = 10^48.
    # P^2 ~ 10^96.
    # Double precision max ~ 10^308. Safe.
    
    P_vals = np.prod(diff**2, axis=0)
    
    # Weights for the integral int_0^inf e^-x f(x) dx are w_gl
    # I0 = sum w_gl * P_vals
    # I1 = sum w_gl * x_gl * P_vals
    
    I0 = np.sum(w_gl * P_vals)
    I1 = np.sum(w_gl * x_gl * P_vals)
    
    if I0 <= 0:
        return 1e10, 0, 0
        
    ratio = I1 / I0
    return ratio, I0, I1

def _objective(roots):
    """
    Objective function to minimize.
    """
    if np.any(roots <= 0):
        return 1e10
    ratio, _, _ = _compute_ratios(roots)
    return ratio

def construct():
    start_time = time.time()
    # Keep a safety margin of 20s before the 600s limit
    deadline = start_time + 580.0
    
    # Initial guess: seed values
    r0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Bounds for roots
    # Roots should be positive. 
    # Upper bound: if roots are too large, the polynomial is shifted right.
    # The integral is dominated by the region where P is large.
    # Let's set a reasonable upper bound, e.g., 100.
    bounds = [(0.1, 100.0) for _ in range(12)]
    
    best_r = r0.copy()
    best_val = _objective(best_r)
    
    # Optimization using L-BFGS-B
    # This method handles bounds well.
    try:
        res = opt.minimize(
            _objective, 
            best_r, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={
                'maxiter': 1000, 
                'ftol': 1e-14, 
                'gtol': 1e-8, 
                'disp': False,
                'maxfun': 5000
            }
        )
        
        if res.success and res.fun < best_val:
            best_r = res.x
            best_val = res.fun
        elif res.fun < best_val:
            best_r = res.x
            best_val = res.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return [float(x) for x in np.sort(best_r)]
        
    # Try a local refinement with Nelder-Mead if time allows
    # Nelder-Mead doesn't use gradients, might escape local minima if L-BFGS-B got stuck
    # But L-BFGS-B is usually sufficient for this smooth landscape.
    # Let's do a quick polish with COBYLA or just return.
    
    # Sort the roots for consistent output
    best_r_sorted = np.sort(best_r)
    
    return [float(x) for x in best_r_sorted]

# EVOLVE-BLOCK-END
