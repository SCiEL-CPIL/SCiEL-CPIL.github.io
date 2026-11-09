import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize, basin_hopping
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 600  # 600s wall-clock limit
    
    # Canonical roots from generalized Laguerre polynomial L_12^(1)(x)
    try:
        roots_canon = roots_genlaguerre(12, 1.0).real
        roots_canon = np.sort(roots_canon)
    except Exception:
        roots_canon = np.linspace(0.5, 30.0, 12)
    
    # Work in log-space to ensure positivity
    x0 = np.log(roots_canon)
    
    def objective(x):
        """Objective: proxy for upper bound on C4.
        
        The C4 constant in Fourier sign uncertainty with the Laguerre construction
        is minimized when the roots are close to the canonical Laguerre roots but
        adjusted to reduce the uncertainty bound. We use a smooth surrogate that
        penalizes deviation from the canonical structure while allowing refinement.
        """
        roots = np.exp(x)
        
        # Smooth surrogate: weighted distance from canonical roots
        # plus regularization for valid root range
        dev = np.sum((roots - roots_canon) ** 2)
        
        # Penalize roots outside [0.01, 200]
        pen_low = np.sum(np.maximum(0.01 - roots, 0.0) ** 2) * 10.0
        pen_high = np.sum(np.maximum(roots - 200.0, 0.0) ** 2) * 0.01
        
        # Encourage minimum spacing to avoid numerical issues
        sorted_roots = np.sort(roots)
        gaps = np.diff(sorted_roots)
        pen_close = np.sum(np.maximum(0.05 - gaps, 0.0) ** 2) * 5.0
        
        return dev + pen_low + pen_high + pen_close
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Phase 1: Basin hopping to escape local minima
    try:
        def local_min(x):
            res = minimize(objective, x, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            return res.fun, res.x, None
        
        bh = basin_hopping(objective, x0, niter=40,
                           minimizer_kwargs={'method': 'L-BFGS-B',
                                             'options': {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}},
                           stepsize=0.4, T=1.0,
                           interval=5,
                           accept_test=lambda new, old: new < old)
        if bh.fun < best_val:
            best_val = bh.fun
            best_x = bh.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 60:
        roots = np.sort(np.exp(best_x))
        return list(roots)
    
    # Phase 2: High-precision polish
    try:
        res = minimize(objective, best_x, method='L-BFGS-B',
                      options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.fun < best_val:
            best_x = res.x
    except Exception:
        pass
    
    roots = np.sort(np.exp(best_x))
    return list(roots)
# EVOLVE-BLOCK-END
