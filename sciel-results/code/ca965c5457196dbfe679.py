"""
Optimizer using Multi-Start Basin Hopping with AOCC-aware budget allocation.
Strategy:
1. Global Phase: Use Latin Hypercube Sampling (LHS) and random restarts to find good basins.
2. Local Phase: Use Nelder-Mead (derivative-free, robust to ridges) for fast local convergence.
3. Dynamic Allocation: Monitor progress. If stagnating, switch to global exploration.
   If converging, switch to local refinement.
4. AOCC Optimization: Prioritize early rapid reduction in error to maximize the Area Under the Curve.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            try:
                f = func(x)
                calls_made += 1
                return f
            except:
                return np.inf

        # Phase 1: Initial Global Exploration (LHS + Random)
        # Allocate ~20% of budget to initial global search to find a decent basin
        initial_budget = max(10, int(budget * 0.2))
        
        # Generate LHS points
        def lhs_points(n, dim):
            indices = np.arange(n)
            points = (indices[:, None] + rng.random((n, 1))) / n
            points = points * 10 - 5  # Map to [-5, 5]
            rng.shuffle(points, axis=0)
            return points

        # Start with a mix of LHS and random
        n_lhs = max(5, initial_budget // 2)
        n_rand = initial_budget - n_lhs
        
        points = []
        if n_lhs > 0:
            points.extend(lhs_points(n_lhs, dim))
        for _ in range(n_rand):
            points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate initial points
        for p in points:
            if time.time() > deadline or calls_made >= budget:
                break
            f = safe_call(p)
            if f < np.inf:
                update_best(p, f)
        
        if best_x is None:
            # Fallback if all failed (should not happen)
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f >= np.inf:
                best_f = 0.0

        # Phase 2: Adaptive Basin Hopping
        # We alternate between local refinement and global jumps.
        # The "jump" size adapts based on recent performance.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Track history for stagnation detection
        recent_improvements = []
        stagnation_threshold = 0.0
        jump_size = 1.0  # Initial jump scale
        
        # Local optimizer settings
        nm_options = {'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
        
        while calls_made < budget and time.time() < deadline:
            # 1. Local Refinement (Nelder-Mead)
            # We wrap the function to count calls and respect budget
            local_calls = [0]
            def local_func(x):
                if local_calls[0] >= budget - calls_made:
                    return np.inf
                f = safe_call(x)
                local_calls[0] += 1
                return f
            
            # Set a cap on local function calls to prevent one NM run from eating all budget
            max_local_calls = min(50, budget - calls_made)
            nm_options['maxiter'] = max_local_calls // 2 # Rough estimate
            
            try:
                res = minimize(local_func, current_x, method='Nelder-Mead', options=nm_options)
                if res.success or res.fun < current_f:
                    new_x = res.x
                    new_f = res.fun
                    if new_f < current_f:
                        # Improvement found
                        current_x = new_x.copy()
                        current_f = new_f
                        update_best(new_x, new_f)
                        recent_improvements.append(1)
                        # Reduce jump size slightly as we are getting closer
                        jump_size *= 0.8
                    else:
                        recent_improvements.append(0)
                else:
                    recent_improvements.append(0)
            except Exception:
                recent_improvements.append(0)
            
            # Keep only last 10 improvements for stagnation check
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)
            
            # 2. Global Jump / Restart Logic
            # If we are stagnating (no improvement in last 10 local runs) or if we have budget to spare
            # and haven't improved globally in a while, do a global jump.
            
            # Check for global stagnation: if best_f hasn't improved significantly
            # and we are still far from the potential optimum (heuristic: if f is still large)
            
            # Simple heuristic: If recent_improvements has few 1s, we are stuck locally.
            if sum(recent_improvements) < 3:
                # Stagnating locally. Do a global jump.
                # Jump size adapts: if we've been stuck for a while, jump bigger.
                # If we just started, jump smaller to stay near the best.
                
                # Adaptive jump: start large, shrink if we find improvements, grow if stuck
                if sum(recent_improvements) == 0:
                    jump_size *= 1.5
                else:
                    jump_size *= 1.1
                
                jump_size = min(jump_size, 4.0) # Cap jump size
                
                # Generate a new point around the current best
                # Use a mixture of local perturbations and fully random restarts
                if rng.random() < 0.3:
                    # Fully random restart to escape deep local minima
                    new_start = rng.uniform(-5, 5, dim)
                else:
                    # Perturb the current best
                    new_start = current_x + rng.normal(0, jump_size, dim)
                    new_start = np.clip(new_start, -5, 5)
                
                # Evaluate the new start
                f_new_start = safe_call(new_start)
                
                if f_new_start < current_f:
                    # New global best found!
                    current_x = new_start.copy()
                    current_f = f_new_start
                    update_best(new_start, f_new_start)
                    # Reset stagnation counter
                    recent_improvements = [1] * len(recent_improvements)
                    # Reduce jump size as we found a better basin
                    jump_size *= 0.7
                else:
                    # Keep current_x as the base for local search, but maybe nudge it
                    # If the jump was bad, don't change the base for local search
                    # unless we are completely stuck.
                    pass
            
            # Check if we should force a restart if budget is low and we are stuck
            if calls_made > budget * 0.8 and sum(recent_improvements) == 0:
                # Force a random restart to try and find a better basin
                new_start = rng.uniform(-5, 5, dim)
                f_new_start = safe_call(new_start)
                if f_new_start < current_f:
                    current_x = new_start.copy()
                    current_f = f_new_start
                    update_best(new_start, f_new_start)

        return best_x, best_f
