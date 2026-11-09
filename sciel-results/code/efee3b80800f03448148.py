"""Refined uncertainty inequality: Cohn-Goncalves Laguerre construction."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import roots_laguerre, eval_genlaguerre
from scipy.optimize import brentq

_START_TIME = time.time()
_DEADLINE = 600.0  # evaluator kills at 630s, keep 30s safety margin

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def _score_roots(roots):
    """
    Compute the upper bound C4 for a given set of 12 positive double-root locations.
    The 'score' is the upper bound on C4 (lower is better).
    
    For the Cohn-Goncalves Laguerre construction, the upper bound C4 is determined
    by the root configuration. The bound is related to the last sign change of the
    Fourier transform, which depends on the largest root and the root distribution.
    
    The most direct interpretation: C4 = 4 * r_max where r_max is the largest root.
    Smaller roots yield a tighter (smaller) bound.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(np.sort(roots)) < 1e-12):
        return float('inf')
    
    roots_sorted = np.sort(roots)
    return 4.0 * float(roots_sorted[-1])

def _compute_laguerre_12_1_roots():
    """
    Compute the 12 positive real roots of the generalized Laguerre polynomial L_12^1(x).
    Uses scipy.special.eval_genlaguerre for accurate evaluation and brentq for refinement.
    """
    n = 12
    alpha = 1
    
    # L_12^1(x) has 12 positive real roots.
    # The largest root is approximately 4n + 2alpha + 2 = 48 + 2 + 2 = 52.
    # We scan a range [0, 60] to be safe.
    x_max = 60.0
    x_min = 1e-10
    
    # Create a grid to find sign changes
    # 10000 points over [0, 60] is sufficient to isolate 12 roots
    x_grid = np.linspace(x_min, x_max, 10000)
    
    # Evaluate L_12^1(x) on the grid
    # eval_genlaguerre(n, alpha, x)
    vals = eval_genlaguerre(n, alpha, x_grid)
    
    # Find intervals where sign changes occur
    sign_changes = []
    for i in range(len(x_grid) - 1):
        if vals[i] * vals[i+1] < 0:
            sign_changes.append((x_grid[i], x_grid[i+1]))
    
    roots = []
    for a, b in sign_changes:
        try:
            # Refine the root using Brent's method
            root = brentq(lambda x: eval_genlaguerre(n, alpha, x), a, b, xtol=1e-14, rtol=1e-14)
            if root > 0:
                roots.append(root)
        except Exception:
            pass
    
    roots = np.array(roots)
    roots = np.sort(roots)
    
    # Verify we have exactly 12 positive roots
    if len(roots) == 12 and np.all(roots > 0):
        return roots
    
    # Fallback: Use polynomial root finding if grid method fails
    try:
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            coeffs[k] = (-1)**k * math.comb(n + alpha, n - k) / math.factorial(k)
        coeffs_reversed = coeffs[::-1]
        roots_poly = np.roots(coeffs_reversed)
        roots_real = np.array([r.real for r in roots_poly if abs(r.imag) < 1e-10 and r.real > 0])
        roots_real = np.sort(roots_real)
        if len(roots_real) == 12:
            return roots_real
    except Exception:
        pass
        
    # Ultimate fallback: zeros of L_12^0(x) scaled by 1 (better than seed's scale 4)
    # Roots of L_12^0 are slightly smaller than L_12^1, but still better than 4x scale
    return roots_laguerre(12)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    try:
        if _remaining() < 5:
            # Fallback: return natural Laguerre roots
            roots = _compute_laguerre_12_1_roots()
            return [float(r) for r in roots]
        
        # Compute the roots of L_12^1(x)
        # These are the natural double-root locations for the Cohn-Goncalves construction.
        # Using the unscaled roots (scale 1) minimizes the bound compared to the seed (scale 4).
        roots = _compute_laguerre_12_1_roots()
        
        if len(roots) != 12 or np.any(roots <= 0):
            # Fallback to L_12^0 zeros if computation failed
            roots = roots_laguerre(12)
        
        # Verify roots are positive and distinct
        roots_sorted = np.sort(roots)
        if np.any(roots_sorted <= 0) or np.any(np.diff(roots_sorted) < 1e-12):
            # Use fallback
            roots_sorted = roots_laguerre(12)
        
        return [float(r) for r in roots_sorted]
            
    except Exception:
        # Ultimate fallback: small positive roots
        if _remaining() < 2:
            return [0.5 * (i + 1) for i in range(12)]
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
