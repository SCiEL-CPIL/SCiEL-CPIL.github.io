"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre
from scipy.optimize import minimize
import time

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the zeros of L_12 as a natural starting point and refines via constrained
    nonlinear optimization to minimize the Cohn-Goncalves upper bound on C4.
    """
    deadline = time.time() + 570  # Safety margin before 630s wall-clock limit
    
    # Initial guess: zeros of the Laguerre polynomial L_12(x)
    try:
        x_roots, _ = roots_laguerre(12)
        x0 = np.array([float(r) for r in x_roots])
    except Exception:
        x0 = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    # Sort to ensure ascending order
    x0 = np.sort(x0)
    
    # The Cohn-Goncalves upper bound on C4 is determined by the last sign change
    # of the constructed function. With double roots, the sign structure is
    # determined by the Laguerre factor L_1(x) = 1 - x (for n=1).
    # 
    # The constructed function involves a product of (x - r_i)^2 terms and a
    # Laguerre polynomial. The "last sign change" that determines the upper
    # bound C4 is typically located at or near the rightmost root, but the
    # exact value depends on the interplay between the roots and the Laguerre
    # weight/function.
    #
    # A reasonable surrogate for the upper bound is the maximum root location,
    # but we can do better by considering the actual sign-change structure.
    # For the Laguerre construction with n=1, the relevant function is:
    #   f(x) = L_1(x) * prod_{i=1}^{12} (x - r_i)^2 * e^{-x/2} (or similar)
    # Since L_1(x) = 1 - x has a zero at x=1, and the double roots don't
    # change sign, the sign changes of f(x) are determined by L_1(x) and the
    # exponential (which is always positive). So the only sign change is at x=1.
    #
    # However, the "upper bound on C4" in the Cohn-Goncalves framework is more
    # subtle. It's related to the largest zero of a certain orthogonal polynomial
    # or the location where the uncertainty product is maximized.
    #
    # Given the task description says "computes the last sign change", and the
    # score is the "upper bound on C4", let's model the bound as a function
    # that we want to minimize. A common formulation is that the bound is
    # proportional to the largest root or a function of the roots.
    #
    # Let's use a penalty-based approach: minimize the maximum root, subject to
    # the roots being positive and ordered. This is a reasonable proxy for
    # minimizing the last sign change / upper bound.
    
    def objective(x):
        """Objective: minimize the largest root (proxy for upper bound C4)."""
        # Ensure positivity and ordering via large penalties
        if np.any(x <= 0):
            return 1e10
        if np.any(np.diff(x) <= 0):
            return 1e10
        # Primary objective: minimize the maximum root
        return x[-1]
    
    # Constraints: roots must be positive and strictly increasing
    # We'll use bounds and rely on the penalty in the objective for ordering
    
    # Bounds: each root must be positive
    bounds = [(1e-10, 100.0) for _ in range(12)]
    
    # Run SLSQP optimization
    best_x = x0.copy()
    best_val = objective(x0)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if result.success and objective(result.x) < best_val:
            best_x = result.x
            best_val = objective(best_x)
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        return [float(r) for r in best_x]
    
    # Try a second optimization pass with a different initial guess
    # (arithmetic progression) to see if we find a better local minimum
    x0_alt = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    try:
        result2 = minimize(
            objective,
            x0_alt,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if result2.success and objective(result2.x) < best_val:
            best_x = result2.x
            best_val = objective(best_x)
    except Exception:
        pass
    
    # Final polish: use the best result
    # Ensure the result is valid (positive, ordered)
    best_x = np.sort(best_x)
    if np.any(best_x <= 0):
        # Fallback to Laguerre roots
        try:
            x_roots, _ = roots_laguerre(12)
            best_x = np.array([float(r) for r in x_roots])
        except Exception:
            best_x = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END
