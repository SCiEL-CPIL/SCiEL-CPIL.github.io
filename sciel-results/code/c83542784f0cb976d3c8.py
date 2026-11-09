"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with coarse initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget or time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return fx
            except Exception:
                return np.inf

        # Phase 1: Coarse initialization
        # Use a small set of diverse starting points
        # 1. Origin
        # 2. Corners of the box (subset)
        # 3. Random points
        
        start_points = []
        start_points.append(np.zeros(d))
        
        # Add some structured points
        # Midpoints of axes
        for i in range(d):
            pos = np.zeros(d); pos[i] = 4.0
            neg = np.zeros(d); neg[i] = -4.0
            start_points.append(pos)
            start_points.append(neg)
            
        # Add random points
        rng = np.random.default_rng(42)
        for _ in range(20):
            start_points.append(rng.uniform(-4, 4, d))

        # Evaluate start points
        for x in start_points:
            if call_count >= budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # Phase 2: Multi-start Nelder-Mead Refinement
        # Run Nelder-Mead from the top few best starting points
        # We will use a fixed budget for each local search
        
        # Sort start points by function value (we need to store them)
        # Since we didn't store all fx values in a list, let's just use the best few found
        # Or simpler: Just run NM from the current best_x, and maybe a couple of other good ones if we kept track.
        # To keep it simple and robust, we'll run NM from the global best.
        # But multi-start is better. Let's store the top 5 start points.
        
        # Re-evaluate or store? We already evaluated them. 
        # Let's just pick the best 3 unique regions from the start points.
        # Actually, we can just run NM from the best_x found so far.
        # To improve, we can run NM from a few other promising points.
        
        # Let's identify top 3 distinct starting points from the list we just evaluated.
        # We need to track (fx, x) pairs.
        
        # Let's refactor Phase 1 to store results.
        # But we already ran it. 
        # Let's just run NM from best_x.
        # If we have budget left, we can run another NM from a random point near best_x?
        
        # Better approach: 
        # 1. Coarse search (done)
        # 2. Local refinement using Nelder-Mead from the best point.
        # 3. If budget remains, restart NM from a perturbed best point or another good point.
        
        # Define a local search function
        def local_search(x0, max_evals):
            nonlocal best_x, best_f, call_count
            
            # Nelder-Mead parameters
            options = {
                'maxiter': max_evals,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options=options
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Run local search from the best point found in Phase 1
        # Allocate most of the remaining budget to this
        remaining_budget = budget - call_count
        if remaining_budget > 10:
            local_search(best_x, remaining_budget)

        # If we still have budget and time, try a second local search from a perturbed point
        if call_count < budget - 10 and time.time() < deadline:
            # Perturb the best point
            perturbation = 0.5 * np.ones(d)
            x2 = best_x + perturbation * np.random.default_rng(123).choice([-1, 1], d)
            x2 = np.clip(x2, -5, 5)
            remaining_budget = budget - call_count
            if remaining_budget > 10:
                local_search(x2, remaining_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
