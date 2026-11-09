"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre
import time

def _compute_bound(roots):
    """Compute the upper bound on C4 for given root locations.
    
    For the Cohn-Goncalves construction with m=12 double roots,
    the bound is related to 1/sqrt(product of roots).
    """
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return float('inf')
    # Use log-space to avoid overflow/underflow
    log_prod = np.sum(np.log(roots))
    # bound = exp(-0.5 * log_prod)
    # If log_prod is very large, bound is very small.
    return np.exp(-0.5 * log_prod)

def construct():
    """Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Returns list of 12 positive double-root locations.
    Score is the upper bound on C4 (lower is better).
    """
    deadline = time.time() + 590.0  # Safety margin before 600s limit
    
    m = 12
    best_roots = None
    best_score = float('inf')
    
    # We hypothesize that the score improves (value decreases) as the product of roots increases.
    # The product of roots of L_m^(alpha)(x) scales as alpha^m for large alpha.
    # Thus, we want to maximize alpha.
    
    # Test a range of alpha values, focusing on large values.
    # Start with some standard values, then go very large.
    alpha_candidates = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    
    # Also try to fill in gaps logarithmically if time permits
    # But first, test the extreme values
    
    for alpha in alpha_candidates:
        if time.time() > deadline - 10:
            break
        
        try:
            # Compute roots of generalized Laguerre polynomial L_m^(alpha)
            roots, _ = roots_genlaguerre(m, alpha=alpha)
            roots = np.sort(roots)
            
            # Validate roots
            if not np.all(np.isfinite(roots)):
                continue
            if not np.all(roots > 0):
                continue
            
            # Compute score proxy
            score = _compute_bound(roots)
            
            if score < best_score:
                best_score = score
                best_roots = roots.copy()
                
        except Exception:
            # If roots_genlaguerre fails for very large alpha, skip
            continue

    # If no valid roots were found (should not happen), fall back to alpha=0
    if best_roots is None:
        try:
            roots, _ = roots_genlaguerre(m, alpha=0)
            best_roots = np.sort(roots)
        except Exception:
            # Last resort fallback
            best_roots = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    return list(best_roots)
# EVOLVE-BLOCK-END
