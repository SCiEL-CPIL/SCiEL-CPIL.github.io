"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial.laguerre import laggauss

_START_TIME = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _deadline_reached():
    return (time.time() - _START_TIME) >= _DEADLINE

def _objective_neg_log_product_gaps(log_roots):
    """
    Objective to maximize the product of gaps between consecutive roots.
    This is equivalent to maximizing the Vandermonde determinant,
    which is a standard criterion for optimal node placement in
    polynomial approximation and uncertainty bounds.
    """
    if _deadline_reached():
        return 1e10
    roots = np.exp(log_roots)
    # Ensure roots are sorted (exp of sorted log_roots is sorted)
    # Calculate gaps
    gaps = np.diff(roots)
    if np.any(gaps <= 0):
        return 1e10
    # Maximize sum of log(gaps) -> Minimize negative sum
    return -np.sum(np.log(gaps))

def _construct_core():
    # Get initial guess: zeros of Laguerre polynomial L_12
    # laggauss(n) returns nodes (zeros) and weights
    nodes, _ = laggauss(12)
    
    # Ensure nodes are positive (Laguerre zeros are positive)
    if np.any(nodes <= 0):
        nodes = np.maximum(nodes, 1e-6)
        
    log_nodes = np.log(nodes)
    
    # Define bounds for log_roots to keep roots positive and within a reasonable range
    # The seed went up to 48. Let's allow up to 100.
    # Lower bound: 0.1 -> log(0.1)
    # Upper bound: 100 -> log(100)
    bounds = [(np.log(0.1), np.log(100.0)) for _ in range(12)]
    
    # Optimize
    try:
        res = minimize(
            _objective_neg_log_product_gaps, 
            log_nodes, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500}
        )
        if res.success and not _deadline_reached():
            opt_roots = np.exp(res.x)
            # Ensure strictly increasing
            if np.all(np.diff(opt_roots) > 0):
                return opt_roots.tolist()
    except Exception:
        pass
    
    # Fallback to Laguerre nodes if optimization fails or times out
    return nodes.tolist()

def construct():
    if _deadline_reached():
        # Emergency return: seed-like or simple valid roots
        return [4.0 * (i + 1) for i in range(12)]
    
    roots = _construct_core()
    
    # Final validation
    if len(roots) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    if any(r <= 0 for r in roots):
        return [4.0 * (i + 1) for i in range(12)]
    if any(roots[i+1] <= roots[i] for i in range(11)):
        return [4.0 * (i + 1) for i in range(12)]
        
    return roots
# EVOLVE-BLOCK-END
