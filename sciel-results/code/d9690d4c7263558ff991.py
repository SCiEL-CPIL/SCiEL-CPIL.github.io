"""Refined uncertainty inequality: tune Laguerre double roots for C4 upper bound."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START = time.time()
DEADLINE = 630.0 - 30.0  # 600s wall clock, keep 30s safety margin

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _laguerre_combination_norm_squared(roots):
    """
    Compute the squared norm of the Laguerre combination L(x) = prod_{i=1}^{12} (x - r_i)^2
    in the weighted L2 space with weight x^1 e^{-x} on [0, infinity).
    
    The norm squared is integral_0^inf |L(x)|^2 * x * e^{-x} dx.
    
    For m=12, n=1: the Laguerre polynomial basis is orthogonal with weight x^n e^{-x}.
    We express L(x) in the monomial basis, then convert to Laguerre basis to get coefficients,
    and use orthogonality to compute the norm.
    
    Actually, a more direct approach: expand prod (x - r_i)^2 as a polynomial, then
    compute integral_0^inf p(x) * x * e^{-x} dx where p(x) = [prod (x-r_i)^2]^2.
    
    Wait, the norm is of the Laguerre combination itself, not squared. Let me reconsider.
    
    The Cohn-Goncalves construction: we have a polynomial P(x) = prod_{i=1}^m (x - r_i) where
    r_i are positive roots. The "Laguerre combination" refers to expressing this in terms of
    Laguerre polynomials. The uncertainty bound involves the norm of this polynomial in the
    Laguerre-weighted space.
    
    For the Fourier sign uncertainty, the key quantity is the L2 norm of the polynomial
    in the weighted space. With m=12 double roots, P(x) = prod_{i=1}^{12} (x - r_i)^2.
    
    The norm squared = integral_0^inf |P(x)|^2 x e^{-x} dx.
    
    To compute this, expand P(x)^2 as a polynomial q(x) = sum_{k=0}^{24} a_k x^k,
    then integral_0^inf x^k * x * e^{-x} dx = integral_0^inf x^{k+1} e^{-x} dx = Gamma(k+2) = (k+1)!.
    
    So norm_squared = sum_{k=0}^{24} a_k * (k+1)!.
    """
    # Build the polynomial prod_{i=1}^{12} (x - r_i)^2
    # Start with 1
    poly = np.array([1.0])  # coefficients from highest degree to lowest? Let's use low to high.
    # Actually, let's use numpy's poly1d-like approach with arrays [a0, a1, ..., an] for sum a_i x^i
    
    poly = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r x + r^2
        # poly has degree d, new poly has degree d+2
        new_poly = np.zeros(len(poly) + 2)
        for i in range(len(poly)):
            new_poly[i] += poly[i] * r*r       # r^2 * x^i
            new_poly[i+1] += poly[i] * (-2*r)   # -2r * x^{i+1}
            new_poly[i+2] += poly[i] * 1.0       # 1 * x^{i+2}
        poly = new_poly
    
    # Now poly[k] is coefficient of x^k in P(x)
    # P(x)^2: square the polynomial
    # P(x) has degree 24, so P(x)^2 has degree 48
    p2 = np.convolve(poly, poly)
    
    # norm_squared = sum_{k=0}^{48} p2[k] * (k+1)!
    # Compute factorials up to 49!
    norm_sq = 0.0
    for k in range(len(p2)):
        # (k+1)! can be huge; use log space or just compute directly
        # For k up to 48, (k+1)! is at most 49! ~ 6e62, which fits in float64
        fact = 1.0
        for j in range(1, k+2):
            fact *= j
        norm_sq += p2[k] * fact
    
    return norm_sq

def _objective(vars_6):
    """
    vars_6: 6 positive variables, roots are vars_6^2 (squared to enforce positivity)
    Returns the squared norm (lower is better for the upper bound).
    """
    roots = vars_6 ** 2
    return _laguerre_combination_norm_squared(roots)

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre combination.
    """
    if _time_left() < 10.0:
        # Not enough time, return a reasonable default
        return [4.0 * (i + 1) for i in range(12)]
    
    np.random.seed(42)
    
    best_vars = None
    best_val = float('inf')
    
    # Initial seed
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_vars = np.sqrt(seed_roots)
    best_vars = seed_vars.copy()
    best_val = _objective(seed_vars)
    
    # Try DE with a reasonable budget
    # Each root is in [0.1, 200] roughly, so vars in [sqrt(0.1), sqrt(200)] ~ [0.316, 14.14]
    bounds = [(0.1, 50.0) for _ in range(6)]
    
    # Check time before DE
    if _time_left() > 60.0:
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.success or _time_left() > 10.0:
                cand_val = de_result.fun
                cand_vars = de_result.x
                if cand_val < best_val:
                    best_val = cand_val
                    best_vars = cand_vars.copy()
        except Exception:
            pass
    
    # Local polish with multiple starts
    if _time_left() > 30.0:
        starts = [best_vars.copy()]
        # Add some perturbed starts
        for _ in range(5):
            pert = best_vars * (1 + 0.1 * np.random.randn(6))
            pert = np.clip(pert, 0.1, 50.0)
            starts.append(pert)
        # Add random starts
        for _ in range(3):
            rand_vars = np.random.uniform(0.5, 10.0, size=6)
            starts.append(rand_vars)
        
        for start in starts:
            if _time_left() < 15.0:
                break
            try:
                res = minimize(
                    _objective,
                    x0=start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_vars = res.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if _time_left() > 15.0:
        try:
            res = minimize(
                _objective,
                x0=best_vars,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_vars = res.x.copy()
        except Exception:
            pass
    
    # Extract roots
    roots = best_vars ** 2
    # Ensure positivity
    roots = np.maximum(roots, 1e-10)
    
    return [float(r) for r in roots]
