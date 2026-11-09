"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    def time_check():
        return time.time() > deadline - 10.0
    
    # Initial guess: Laguerre zeros
    try:
        roots_init, _ = roots_laguerre(12)
        x0 = np.array(roots_init, dtype=float)
    except Exception:
        x0 = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    # Objective: For the Cohn-Goncalves Laguerre construction with m=12, n=1,
    # the upper bound on C4 is related to the norm or extremal value of the
    # constructed polynomial. A natural candidate is the product of roots
    # (constant term of the monic polynomial), or a weighted sum.
    # 
    # Let's try: the bound involves the value of the Laguerre-weighted integral
    # of the squared polynomial, or the maximum of |P(x)| * w(x) over x >= 0.
    # 
    # A simpler structural choice: minimize the largest root (which controls
    # the "last sign change" location) subject to roots being positive and ordered.
    # This is a structural change: instead of fixed Laguerre zeros, we optimize
    # to push the last root as far left as possible while maintaining validity.
    
    def objective(x):
        """Minimize the largest root location (last sign change proxy)."""
        return x[-1]
    
    # Constraints: roots must be positive and strictly increasing
    # x_i > 0 for all i, and x_{i+1} > x_i for all i
    constraints = []
    # Positivity: x_i > 0.1 (small margin)
    for i in range(12):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 0.1})
    # Ordering: x_{i+1} - x_i > 0.01
    for i in range(11):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 0.01})
    
    # Bounds: each root in [0.1, 200]
    bounds = [(0.1, 200.0)] * 12
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try SLSQP optimization
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        if result.success and result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time_check():
        return [float(r) for r in best_x]
    
    # Try a second optimization with a different objective: minimize sum of roots
    def objective2(x):
        """Minimize sum of roots."""
        return np.sum(x)
    
    try:
        result2 = minimize(
            objective2, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        if result2.success:
            # Compare: we want the one that gives a better (lower) upper bound
            # Without knowing the exact functional, pick the one with smaller max root
            if result2.x[-1] < best_x[-1]:
                best_x = result2.x
    except Exception:
        pass
    
    if time_check():
        return [float(r) for r in best_x]
    
    # Try minimizing the product of roots
    def objective3(x):
        """Minimize log of product of roots (sum of logs)."""
        return np.sum(np.log(x))
    
    try:
        result3 = minimize(
            objective3, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        if result3.success and result3.x[-1] < best_x[-1]:
            best_x = result3.x
    except Exception:
        pass
    
    # Ensure final result is valid
    if any(r <= 0 for r in best_x):
        best_x = x0.copy()
    
    # Sort to ensure ordering
    best_x = np.sort(best_x)
    
    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END
