"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def find_root_system(F, x0):
    """
    Finds a root x for F(x) = 0 using global optimization of the squared residual.
    
    Strategy:
    1. Define objective function as ||F(x)||^2.
    2. Use BFGS (quasi-Newton) for local minimization, which is robust and efficient.
    3. Perform multi-start optimization from a grid of initial points around x0 
       and some global candidates to avoid local minima.
    4. Track the best solution found (lowest residual) and return it if within tolerance.
    5. Strict time management to avoid timeout.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def objective(x):
        """
        Objective function: sum of squares of F(x).
        Returns inf if F(x) is invalid or non-finite.
        """
        try:
            fx = np.asarray(F(x), dtype=float)
            if not np.all(np.isfinite(fx)):
                return np.inf
            return np.sum(fx * fx)
        except Exception:
            return np.inf

    def is_good(x):
        """
        Check if x is a valid root: finite x, finite F(x), ||F(x)|| <= 1e-8.
        """
        if not np.all(np.isfinite(x)):
            return False
        try:
            fx = np.asarray(F(x), dtype=float)
            if not np.all(np.isfinite(fx)):
                return False
            return np.linalg.norm(fx) <= 1e-8
        except Exception:
            return False

    # Ensure x0 is a valid starting point
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # If x0 is already a root, return it
    if best_val <= (1e-8)**2:
        return best_x

    # Generate candidate starting points for multi-start
    candidates = []
    
    # 1. The provided x0
    candidates.append(x0.copy())
    
    # 2. Origin
    candidates.append(np.array([0.0, 0.0]))
    
    # 3. Grid around x0
    # Scale based on magnitude of x0, default to 1.0 if x0 is 0
    scale = max(1.0, np.linalg.norm(x0))
    
    # Small perturbations
    for dx in [-scale, 0.0, scale]:
        for dy in [-scale, 0.0, scale]:
            candidates.append(x0 + np.array([dx, dy]))
            
    # 4. Larger scale perturbations to escape local minima
    for factor in [0.5, 2.0, 5.0]:
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            offset = factor * scale * np.array([np.cos(angle), np.sin(angle)])
            candidates.append(x0 + offset)
            
    # 5. Fixed global candidates (in case x0 is far from root)
    for r in [1.0, 10.0, 100.0]:
        for angle in np.linspace(0, 2*np.pi, 4, endpoint=False):
            candidates.append(r * np.array([np.cos(angle), np.sin(angle)]))

    # Remove duplicates and non-finite candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        if not np.all(np.isfinite(c)):
            continue
        # Round to avoid floating point duplicates in set
        key = tuple(np.round(c, 8))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Sort candidates by initial objective value to try promising ones first?
    # Or just iterate. Sorting requires evaluating all, which costs calls.
    # Let's just iterate, but prioritize x0 and origin first (already done).
    
    # Optimization Loop
    # BFGS is a quasi-Newton method that approximates the Hessian.
    # It is generally robust for smooth functions.
    # We use 'gtol' to stop when gradient is small, but we check residual directly.
    
    for x_start in unique_candidates:
        if time_left() < 1.0:
            break
            
        # Skip if we've already tried a very close point
        # (Optional optimization, but adds complexity. Let's rely on 'seen' set above)
        
        try:
            # Use BFGS
            # maxiter: limit iterations to prevent long hangs on a single start
            # ftol, xtol, gtol: standard tolerances
            res = minimize(
                objective, 
                x_start, 
                method='BFGS', 
                options={'maxiter': 100, 'gtol': 1e-8, 'disp': False}
            )
            
            x_opt = res.x
            val_opt = res.fun
            
            if np.all(np.isfinite(x_opt)) and val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
                
                # Early exit if we found a root
                if best_val <= (1e-8)**2:
                    return best_x
                    
        except Exception:
            continue
            
    # Final check
    if is_good(best_x):
        return best_x
        
    # If we didn't find a root within tolerance, return the best found anyway.
    # The evaluator will mark it as failure if ||F(x)|| > 1e-8, but this is better than crashing.
    return best_x

# EVOLVE-BLOCK-END
