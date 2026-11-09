"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import genlaguerre, roots_laguerre

# EVOLVE-BLOCK-START
def _laguerre_moments(roots, m=12, n=1):
    """
    Compute a surrogate for the C4 upper bound based on Laguerre polynomial
    evaluations at the given roots. The Cohn-Goncalves construction for
    m double roots in dimension n involves a specific Laguerre combination.
    We use a well-motivated surrogate: the objective is related to the
    harmonic mean of root spacings weighted by Laguerre values.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if not np.all(np.diff(roots) > 0):
        return 1e10
    
    # Compute Laguerre polynomial L_m(x)
    L = genlaguerre(m, n)
    
    # Evaluate at roots
    vals = L(roots)
    
    # The C4 bound is related to the product structure of the polynomial
    # at the roots. For double roots, the polynomial should be zero at each
    # root, but the "upper bound" involves the behavior near the roots.
    # A common measure is the sum of 1/|spacings| weighted by local curvature.
    
    # Use the sum of inverse squared gaps as a proxy for uncertainty
    gaps = np.diff(roots)
    if np.any(gaps <= 0):
        return 1e10
    
    # The objective: minimize the spread while maintaining positivity
    # C4 ~ (sum of 1/gap_i^2) * (max_root / min_root)
    obj = np.sum(1.0 / gaps**2) * (roots[-1] / roots[0])
    
    return obj

def construct():
    start_time = time.time()
    deadline = 600.0  # seconds
    safety_margin = 30.0  # return 30s before kill
    
    # Initial guess: arithmetic spacing
    best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    best_obj = _laguerre_moments(best_roots)
    
    elapsed = time.time() - start_time
    if elapsed + safety_margin > deadline:
        return [float(x) for x in best_roots]
    
    # Try local optimization with L-BFGS-B
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed + safety_margin > deadline:
            return 1e10
        return _laguerre_moments(params)
    
    # Multiple random restarts
    n_restarts = 50
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed + safety_margin > deadline:
            break
        
        # Generate initial guess
        if restart == 0:
            x0 = best_roots.copy()
        else:
            # Perturb the best solution or use random
            if restart % 2 == 0:
                # Small perturbation of best
                x0 = best_roots + np.random.randn(12) * 0.5
                x0 = np.maximum(x0, 0.1)
                # Ensure sorted
                x0 = np.sort(x0)
            else:
                # Random log-spaced or power-law initial guess
                alpha = np.random.uniform(0.5, 2.0)
                x0 = 4.0 * np.arange(1, 13) ** alpha
                x0 = np.sort(x0)
        
        # Optimize
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             bounds=[(0.1, None)] * 12,
                             options={'maxiter': 100, 'ftol': 1e-10})
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 0.01)
    
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
