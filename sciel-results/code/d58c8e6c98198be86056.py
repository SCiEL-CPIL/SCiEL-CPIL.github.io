import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Strategy:
    Instead of treating the 12 roots as independent variables, we represent them
    as the nodes of a generalized Gauss-Laguerre quadrature rule for the weight
    w(x) = e^{-x} x^alpha. The roots are the zeros of L_12^(alpha)(x).
    We optimize the single parameter alpha to minimize a proxy for the 
    uncertainty bound C4.
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    m = 12
    # alpha must be > -1 for roots_genlaguerre
    # Scan alpha in a reasonable range
    # Seed is alpha=0.
    # Try a range that allows roots to move significantly.
    alphas = np.linspace(-0.9, 5.0, 1000)
    
    best_score = np.inf
    best_roots = None
    best_alpha = 0.0
    
    def proxy_score(roots):
        """
        Proxy for the upper bound on C4.
        Based on Inspiration 1's successful objective (score 0.9817),
        which rewarded compact, well-packed roots with a specific weighting.
        """
        r = np.asarray(roots, dtype=float)
        # Positivity check (should be guaranteed by Laguerre roots for alpha > -1)
        if np.any(r <= 0):
            return np.inf
        
        # Weighting function from Inspiration 1
        w = np.exp(-0.05 * r)
        
        # Moments
        base = np.sum(w * r)
        spread = np.sum(w * r ** 2)
        
        # Entropy-like packing term
        log_term = np.sum(np.log(r + 1e-8))
        
        obj = base + 0.5 * spread - 0.02 * log_term
        return obj

    # Scan alpha
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            # Get roots of L_12^(alpha)
            roots = roots_genlaguerre(m, alpha).x
            # Ensure sorted and positive
            roots = np.sort(roots)
            if np.any(roots <= 0):
                continue
                
            score = proxy_score(roots)
            
            if score < best_score:
                best_score = score
                best_roots = roots
                best_alpha = alpha
        except Exception:
            continue
            
    # Fallback if no valid alpha found or if best_roots is None
    if best_roots is None:
        # Use standard Laguerre roots (alpha=0)
        best_roots = roots_genlaguerre(m, 0).x
        best_roots = np.sort(best_roots)
    else:
        # Optional: Local polish on the 12 roots starting from the best parametric solution
        # This combines the structural parametric search with the local refinement machinery.
        # We use a simple gradient-free local search (Nelder-Mead) on the 12 roots
        # to see if we can squeeze out more improvement, constrained to stay positive.
        
        from scipy.optimize import minimize
        
        # Define objective for local polish (same proxy)
        def local_obj(r):
            return proxy_score(r)
        
        # Initial guess is the best parametric roots
        x0 = best_roots
        
        # Bounds: strictly positive
        bounds = [(1e-6, None)] * m
        
        # Use Nelder-Mead as it doesn't require gradients and handles the 
        # non-smooth log term reasonably well for small n.
        # Limit iterations to ensure we don't blow the time budget.
        try:
            res = minimize(
                local_obj, 
                x0, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success and res.fun < best_score:
                # Verify roots are still valid (positive, real)
                new_roots = res.x
                if np.all(new_roots > 0):
                    # Re-sort to ensure order
                    new_roots = np.sort(new_roots)
                    best_roots = new_roots
                    best_score = res.fun
        except Exception:
            pass # Keep parametric result if local opt fails

    # Final safety check
    if best_roots is None or len(best_roots) != m:
        best_roots = roots_genlaguerre(m, 0).x
        best_roots = np.sort(best_roots)
        
    # Ensure strict positivity and ordering for return
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(r) for r in best_roots]
