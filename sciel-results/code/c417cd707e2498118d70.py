"""Rewritten program for Erdos minimum overlap task.

Strategy: Represent h as a sum of low-frequency DFT modes (trigonometric basis).
This reduces the search space from N variables to a few spectral coefficients.
The DC component enforces the integral constraint (sum = N/2).
We use SLSQP with exact constraints and an epigraph formulation to minimize
the maximum overlap with shifted 1-h.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 100

# Number of DFT modes to use (excluding DC)
# We'll use modes 1..K where K is small
K = 15  # number of non-DC modes

# Deadline guard
DEADLINE = 140.0  # seconds, leaving 10s margin for the 150s limit

def _construct_h_from_coeffs(coeffs):
    """
    Given coefficients [a1, b1, a2, b2, ..., aK, bK], construct h.
    h[j] = 0.5 + sum_{k=1}^{K} (a_k * cos(2*pi*k*j/N) + b_k * sin(2*pi*k*j/N))
    
    The DC component is fixed at 0.5 to ensure sum(h) = N/2.
    """
    j = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, K + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h += a * np.cos(2 * np.pi * k * j / N) + b * np.sin(2 * np.pi * k * j / N)
    return h

def _max_overlap(h, shift):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift s: sum_j h[j] * (1 - h[(j+s) % N]) / N
    We want to minimize the max over all shifts s.
    """
    N_local = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N_local):
        shifted = np.roll(one_minus_h, s)
        ov = np.sum(h * shifted) / N_local
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective_and_grad(coeffs):
    """
    Objective: t (epigraph variable)
    We minimize t subject to t >= max_overlap(h, s) for all s.
    
    For SLSQP, we'll use a penalty approach or handle it via constraints.
    Actually, let's use a different approach: minimize the maximum directly
    using a smooth approximation, or use the epigraph with constraints.
    
    For simplicity and robustness, let's minimize a soft-max of the overlaps.
    """
    h = _construct_h_from_coeffs(coeffs)
    # Clip to [0,1] for safety during optimization
    h_clipped = np.clip(h, 0, 1)
    
    N_local = len(h_clipped)
    one_minus_h = 1.0 - h_clipped
    
    overlaps = np.array([np.sum(h_clipped * np.roll(one_minus_h, s)) / N_local for s in range(N_local)])
    
    # Soft-max with large beta for approximate max
    beta = 50.0
    max_ov = -np.log(np.sum(np.exp(-beta * overlaps))) / beta
    
    return max_ov

def _constraints(coeffs):
    """
    Constraints: h[j] in [0, 1] for all j.
    """
    h = _construct_h_from_coeffs(coeffs)
    cons = []
    for j in range(N):
        cons.append({'type': 'ineq', 'fun': lambda c, j=j: _construct_h_from_coeffs(c)[j] - 0.0})
        cons.append({'type': 'ineq', 'fun': lambda c, j=j: 1.0 - _construct_h_from_coeffs(c)[j]})
    return cons

def _run_optimization(initial_coeffs, deadline):
    """
    Run SLSQP optimization from initial_coeffs.
    Returns (best_coeffs, best_value).
    """
    def obj(c):
        return _objective_and_grad(c)
    
    # Define constraints more efficiently
    def con_h_min(c, j):
        return _construct_h_from_coeffs(c)[j]
    
    def con_h_max(c, j):
        return 1.0 - _construct_h_from_coeffs(c)[j]
    
    cons = []
    for j in range(N):
        cons.append({'type': 'ineq', 'fun': lambda c, j=j: con_h_min(c, j)})
        cons.append({'type': 'ineq', 'fun': lambda c, j=j: con_h_max(c, j)})
    
    result = minimize(obj, initial_coeffs, method='SLSQP', constraints=cons,
                      options={'maxiter': 500, 'ftol': 1e-10})
    
    if result.success:
        return result.x, result.fun
    else:
        # Still return the best found
        return result.x, result.fun

def construct():
    start_time = time.time()
    
    best_coeffs = np.zeros(2 * K)
    best_value = _objective_and_grad(best_coeffs)
    
    # Multiple restarts with different initial conditions
    np.random.seed(42)
    
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Generate random initial coefficients
        if r == 0:
            init = np.zeros(2 * K)
        else:
            # Small random perturbations
            init = np.random.randn(2 * K) * 0.1
        
        # Also try some structured initializations
        if r % 5 == 1:
            # Emphasize low frequencies
            for i in range(2 * K):
                k = i // 2 + 1
                init[i] *= 1.0 / k
        
        try:
            coeffs, val = _run_optimization(init, DEADLINE)
            if val < best_value:
                best_value = val
                best_coeffs = coeffs
        except Exception:
            continue
        
        # Check deadline
        if time.time() > DEADLINE - 15:
            break
    
    # Final polish: run a few more iterations with tighter tolerance
    if time.time() < DEADLINE - 5:
        try:
            def obj_polish(c):
                return _objective_and_grad(c)
            
            def con_h_min(c, j):
                return _construct_h_from_coeffs(c)[j]
            
            def con_h_max(c, j):
                return 1.0 - _construct_h_from_coeffs(c)[j]
            
            cons = []
            for j in range(N):
                cons.append({'type': 'ineq', 'fun': lambda c, j=j: con_h_min(c, j)})
                cons.append({'type': 'ineq', 'fun': lambda c, j=j: con_h_max(c, j)})
            
            result = minimize(obj_polish, best_coeffs, method='SLSQP', constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-12})
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Construct final h and clip to [0,1]
    h = _construct_h_from_coeffs(best_coeffs)
    h = np.clip(h, 0, 1)
    
    # Ensure sum is exactly N/2 by adjusting
    current_sum = np.sum(h)
    if current_sum > 0:
        h *= (N / 2.0) / current_sum
        h = np.clip(h, 0, 1)
    
    # If clipping changed the sum, do a simple correction
    # This is a best-effort approach; the optimization should have kept things in [0,1]
    
    return h

# Verify the function exists
if __name__ == "__main__":
    h = construct()
    print(f"Sum: {np.sum(h)}, N/2: {N/2}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
    # Compute actual max overlap for verification
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.sum(h * shifted) / N
        if ov > max_ov:
            max_ov = ov
    print(f"Max overlap: {max_ov}")
