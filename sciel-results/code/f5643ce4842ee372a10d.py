"""Constrained optimization approach: minimize r^2 subject to g(x) >= 0 for x >= 0."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x).
    H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
    Returns list where index i is coefficient of x^i.
    """
    # Use recurrence
    h0 = [1.0]
    if n == 0:
        return h0
    h1 = [0.0, 2.0]
    if n == 1:
        return h1
    for k in range(1, n):
        # h2 = 2x * h1 - 2k * h0
        # 2x * h1: shift h1 right by 1 and multiply by 2
        term1 = [0.0] + [2.0 * c for c in h1]
        # 2k * h0
        term2 = [2.0 * k * c for c in h0]
        # h2 = term1 - term2
        max_len = max(len(term1), len(term2))
        h2 = [0.0] * max_len
        for i in range(len(term1)):
            h2[i] += term1[i]
        for i in range(len(term2)):
            h2[i] -= term2[i]
        h0, h1 = h1, h2
    return h1

def _poly_coeffs(c0, c1, c2, c3):
    """Return coefficients of g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    H0 = _hermite_coeffs(0)
    H4 = _hermite_coeffs(4)
    H8 = _hermite_coeffs(8)
    H12 = _hermite_coeffs(12)
    
    max_deg = max(len(H0), len(H4), len(H8), len(H12))
    result = [0.0] * max_deg
    
    for i, c in enumerate(H0):
        result[i] += c0 * c
    for i, c in enumerate(H4):
        result[i] += c1 * c
    for i, c in enumerate(H8):
        result[i] += c2 * c
    for i, c in enumerate(H12):
        result[i] += c3 * c
    
    return result

def _compute_c3(c0, c1, c2):
    """Compute c3 so that g(0) = 0."""
    # H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -1330560.0
    return -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0

def _find_largest_sign_changing_root(coeffs):
    """Find the largest real root where the polynomial changes sign (odd multiplicity).
    coeffs: list of coefficients from lowest to highest degree.
    Returns (root, is_valid) or (None, False).
    """
    # Remove trailing zeros
    while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-15:
        coeffs = coeffs[:-1]
    
    if len(coeffs) <= 1:
        return None, False
    
    # Use numpy to find roots
    # numpy.roots expects highest degree first
    np_coeffs = np.array(coeffs[::-1])
    
    # Check if polynomial is identically zero
    if np.all(np.abs(np_coeffs) < 1e-15):
        return None, False
    
    try:
        roots = np.roots(np_coeffs)
    except Exception:
        return None, False
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    if not real_roots:
        return None, False
    
    # For each real root, check if it's a sign-changing root (odd multiplicity)
    # A root has odd multiplicity if the polynomial changes sign at that root
    best_r = None
    
    for r in real_roots:
        if r <= 0:
            continue
        
        # Check sign change by evaluating at r - eps and r + eps
        eps = 1e-8 * max(1.0, abs(r))
        
        # Evaluate polynomial at r - eps and r + eps
        val_minus = np.polyval(np_coeffs, r - eps)
        val_plus = np.polyval(np_coeffs, r + eps)
        
        # Sign change if val_minus and val_plus have opposite signs
        if val_minus * val_plus < 0:
            if best_r is None or r > best_r:
                best_r = r
    
    if best_r is None:
        return None, False
    
    return best_r, True

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients. Returns (value, valid)."""
    c3 = _compute_c3(c0, c1, c2)
    
    # Build g(x) coefficients
    g_coeffs = _poly_coeffs(c0, c1, c2, c3)
    
    # q(x) = g(x) / x^2, so remove first two coefficients (x^0 and x^1 terms)
    # g(0) = 0 should hold, and g'(0) = 0 should hold for exact division
    if len(g_coeffs) < 3:
        return None, False
    
    # Check that g(0) ≈ 0 and g'(0) ≈ 0
    if abs(g_coeffs[0]) > 1e-8 or abs(g_coeffs[1]) > 1e-8:
        # Try to proceed anyway, but this might indicate numerical issues
        pass
    
    q_coeffs = g_coeffs[2:]  # Remove x^0 and x^1 terms
    
    # Find largest sign-changing root of q
    r, valid = _find_largest_sign_changing_root(q_coeffs)
    if not valid or r is None:
        return None, False
    
    value = r**2 / (2.0 * np.pi)
    return value, True

def _objective(params, penalty_weight=100.0, grid_points=200):
    """Objective function: minimize r^2 + penalty for g(x) < 0 on x >= 0."""
    c0, c1, c2 = params
    c3 = _compute_c3(c0, c1, c2)
    
    g_coeffs = _poly_coeffs(c0, c1, c2, c3)
    
    if len(g_coeffs) < 3:
        return 1e10
    
    q_coeffs = g_coeffs[2:]
    
    # Find largest sign-changing root
    r, valid = _find_largest_sign_changing_root(q_coeffs)
    if not valid or r is None:
        return 1e10
    
    # Penalty for g(x) < 0 on x >= 0
    # Sample g(x) on a grid from 0 to some max
    max_x = max(5.0, 2.0 * r)
    xs = np.linspace(0.0, max_x, grid_points)
    
    # Evaluate g(x) using Horner's method
    # g_coeffs is from lowest to highest degree
    g_vals = np.zeros_like(xs)
    for i, coeff in enumerate(g_coeffs):
        g_vals += coeff * xs**i
    
    # Penalty: sum of negative parts squared
    neg_parts = np.maximum(0.0, -g_vals)
    penalty = penalty_weight * np.sum(neg_parts**2)
    
    return r**2 + penalty

def _objective_simple(params, grid_points=100):
    """Simpler objective: just r^2 with a hard penalty if g goes negative."""
    c0, c1, c2 = params
    c3 = _compute_c3(c0, c1, c2)
    
    g_coeffs = _poly_coeffs(c0, c1, c2, c3)
    
    if len(g_coeffs) < 3:
        return 1e10
    
    q_coeffs = g_coeffs[2:]
    
    r, valid = _find_largest_sign_changing_root(q_coeffs)
    if not valid or r is None:
        return 1e10
    
    # Check if g(x) >= 0 on x >= 0
    max_x = max(5.0, 2.0 * r)
    xs = np.linspace(0.0, max_x, grid_points)
    
    g_vals = np.zeros_like(xs)
    for i, coeff in enumerate(g_coeffs):
        g_vals += coeff * xs**i
    
    if np.min(g_vals) < -1e-6:
        return r**2 + 1000.0 * np.sum(np.maximum(0, -g_vals)**2)
    
    return r**2

def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 200.0  # Leave 10s safety margin
    
    # Start with the seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0], dtype=np.float64)
    best_coeffs = seed.copy()
    best_value, _ = _compute_value(*best_coeffs)
    if best_value is None:
        best_value = float('inf')
    
    # Try Nelder-Mead optimization with penalty
    rng = np.random.RandomState(42)
    
    # Multiple random restarts
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = seed.copy()
        else:
            # Random perturbation around seed
            scale = 0.1 / (restart + 1)
            x0 = seed + rng.normal(0, scale, size=3)
        
        try:
            result = minimize(
                _objective_simple,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < best_value:
                c0, c1, c2 = result.x
                val, valid = _compute_value(c0, c1, c2)
                if valid and val is not None and val < best_value:
                    best_value = val
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Local search refinement around best found
    if time.time() < deadline:
        scales = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        current = best_coeffs.copy()
        current_value = best_value
        
        for scale in scales:
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                delta = rng.normal(0, scale, size=3)
                new_coeffs = current + delta
                
                val, valid = _compute_value(*new_coeffs)
                if val is not None and valid and val < current_value:
                    current = new_coeffs
                    current_value = val
                    if current_value < best_value:
                        best_value = current_value
                        best_coeffs = current.copy()
        
        if time.time() > deadline:
            break
    
    return best_coeffs
# EVOLVE-BLOCK-END
