# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Optimizer combining coordinate-descent warm start with Nelder-Mead refinement.
    
    Strategy:
    1. Initialization: Sample 15 random points to find a promising starting basin.
    2. Warm Start: Use a coordinate descent strategy with multi-scale steps to rapidly
       descend into the local basin. This provides strong early progress, which is
       critical for the AOCC score.
    3. Refinement: Use scipy's Nelder-Mead implementation for robust local convergence.
       Nelder-Mead adapts the shape of the search region, which is often more efficient
       than (1+1)-ES for smooth functions with well-defined basins like BBOB f13.
    4. Budget Management: Monitor function calls and wall-clock time strictly.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            # Ensure x is a numpy array of floats
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization with random sampling
        n_init = 15
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-5.0, 5.0, dim)
            init_points.append(x)
        
        for x in init_points:
            fx = timed_call(x)
            if fx is None:
                break

        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Phase 2: Coordinate Descent Warm Start
        # This phase aims to make rapid progress by exploring the neighborhood of the best point
        # using a series of step sizes, similar to a multi-scale grid search but adaptive.
        
        # We will use a fixed number of steps for this phase to ensure early progress
        # but stop if budget or time runs out.
        
        # Start from the best point found in Phase 1
        x_current = best_x.copy()
        f_current = best_f
        
        # Multi-scale steps: start large, get finer
        # We limit the total number of calls in this phase to ~50-100 to save budget for NM
        max_cd_calls = 100
        cd_calls = 0
        steps = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
        
        for step in steps:
            if cd_calls >= max_cd_calls or calls_made >= self.budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Try to improve along each dimension
            improved = True
            while improved:
                improved = False
                for d in range(dim):
                    if cd_calls >= max_cd_calls or calls_made >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    # Try positive step
                    x_trial = x_current.copy()
                    x_trial[d] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = timed_call(x_trial)
                    if f_trial is None:
                        break
                    
                    if f_trial < f_current:
                        x_current = x_trial
                        f_current = f_trial
                        improved = True
                        cd_calls += 1
                        # Update global best
                        if f_current < best_f:
                            best_f = f_current
                            best_x = x_current.copy()
                        continue # Restart dimension loop if improved? Or just continue to next dim?
                        # Standard CD usually continues to next dim. Let's do that.
                        # But the `improved` flag handles the outer while loop.
                    
                    # Try negative step
                    x_trial = x_current.copy()
                    x_trial[d] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = timed_call(x_trial)
                    if f_trial is None:
                        break
                    
                    if f_trial < f_current:
                        x_current = x_trial
                        f_current = f_trial
                        improved = True
                        cd_calls += 1
                        # Update global best
                        if f_current < best_f:
                            best_f = f_current
                            best_x = x_current.copy()
                
                if f_trial is None:
                    break

        # Phase 3: Nelder-Mead Refinement
        # Use scipy's minimize for robust NM.
        # We need to wrap the function to respect the budget and time limits.
        
        nm_wrapper_calls = [0]
        def nm_func(x):
            nonlocal best_f, best_x
            if calls_made >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline - 2:
                raise Exception("Time limit exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_made += 1
            nm_wrapper_calls[0] += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Run NM from the current best point
        try:
            res = minimize(
                nm_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            # The minimize function updates best_x and best_f via nm_func
        except Exception:
            pass

        # Phase 4: Second NM pass if budget remains, with tighter tolerance
        remaining_budget = self.budget - calls_made
        if remaining_budget > 20 and time.time() < deadline - 5:
            try:
                res2 = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining_budget // 10, 100), 'xatol': 1e-12, 'fatol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
