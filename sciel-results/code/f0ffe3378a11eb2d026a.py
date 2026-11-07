"""Portfolio optimizer for BBOB-style f7 (Step ellipsoidal), dim=5, budget=500."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Brent's Method-based Coordinate Descent optimizer.
    Exploits 1D structure for high-precision convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # 300 s evaluator limit, 15 s margin
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0

        # ---- bookkeeping -------------------------------------------------
        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            v = float(func(x))
            return v

        # ---- Phase 0: cheap initial samples -----------------------------
        # Sample center and axis extremes to find a reasonable starting point
        xs = []
        xs.append(np.full(d, 0.0))
        for i in range(d):
            p = np.full(d, 0.0); p[i] = -4.0; xs.append(p)
            p = np.full(d, 0.0); p[i] =  4.0; xs.append(p)

        best_x = None
        best_f = np.inf
        for x in xs:
            if used >= B or time.time() > deadline:
                break
            v = f(x)
            if v < best_f:
                best_f, best_x = v, x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = f(best_x)

        # ---- Phase 1: Coordinate Descent with Brent's Method ------------
        # Iterate over dimensions, minimizing along each axis using Brent's method.
        # Repeat until budget/time runs out.
        
        # We define a helper to minimize along a specific axis
        def minimize_axis(x_base, axis_idx):
            nonlocal used, deadline
            if used >= B or time.time() > deadline:
                return x_base, best_f
            
            # Define the 1D function to minimize
            def obj_1d(t):
                x_new = x_base.copy()
                x_new[axis_idx] = t
                return f(x_new)
            
            # Determine bounds for the 1D search
            # Start with a wide range, but clipped to box
            lo = max(LO, x_base[axis_idx] - 5.0)
            hi = min(HI, x_base[axis_idx] + 5.0)
            
            # If the range is too small, skip
            if hi - lo < 1e-6:
                return x_base, best_f

            try:
                # Use Brent's method
                # xatol controls the stopping criterion
                # We use a relatively tight tolerance to ensure precision
                result = minimize_scalar(
                    obj_1d, 
                    bounds=(lo, hi), 
                    method='brent', 
                    options={'xatol': 1e-4, 'maxiter': 50}
                )
                
                if result.success and result.fun < best_f:
                    new_x = x_base.copy()
                    new_x[axis_idx] = result.x
                    return new_x, result.fun
            except Exception:
                pass
            
            return x_base, best_f

        # Main loop: cycle through axes
        # We don't know how many full passes we can do, so we just loop
        # until budget or time is exhausted.
        # To prevent infinite loops if no improvement is made, we track
        # the best function value and break if no improvement in a full pass.
        
        while used < B and time.time() < deadline:
            improved = False
            for i in range(d):
                if used >= B or time.time() > deadline:
                    break
                
                new_x, new_f = minimize_axis(best_x, i)
                
                if new_f < best_f - 1e-8:
                    best_x = new_x
                    best_f = new_f
                    improved = True
                elif new_f < best_f:
                    # Tiny improvement, still update to move closer
                    best_x = new_x
                    best_f = new_f
                    improved = True

            if not improved:
                # No improvement in a full pass. 
                # We could try a random restart or tighter bounds, 
                # but for this simple CD, we might just stop or 
                # do a final polish.
                # Let's do a final polish with tighter bounds around best_x
                break

        # ---- Phase 2: Final Polish ---------------------------------------
        # If we have budget left, do a few more passes with tighter bounds
        # around the current best_x to squeeze out more precision.
        if used < B and time.time() < deadline:
            for _ in range(3): # A few extra passes
                if used >= B or time.time() > deadline:
                    break
                for i in range(d):
                    if used >= B or time.time() > deadline:
                        break
                    
                    # Tighter bounds for polish
                    lo = max(LO, best_x[i] - 1.0)
                    hi = min(HI, best_x[i] + 1.0)
                    
                    def obj_1d_polish(t):
                        x_new = best_x.copy()
                        x_new[i] = t
                        return f(x_new)
                    
                    try:
                        result = minimize_scalar(
                            obj_1d_polish, 
                            bounds=(lo, hi), 
                            method='brent', 
                            options={'xatol': 1e-6, 'maxiter': 50}
                        )
                        if result.success and result.fun < best_f:
                            best_x[i] = result.x
                            best_f = result.fun
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
