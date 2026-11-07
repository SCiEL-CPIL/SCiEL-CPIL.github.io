"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Adaptive Local Optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_func(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            if calls_made >= budget:
                return np.inf
            val = func(x)
            calls_made += 1
            return val

        def check_time():
            return time.time() > deadline

        # Phase 1: Global Exploration (Multi-Start)
        # Use Sobol-like or random points to cover the space.
        # Given the "shifted" nature, uniform random is robust.
        # We allocate ~10-15% of budget to exploration.
        explore_budget = max(20, int(budget * 0.15))
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # We want to find the basin of attraction quickly.
        # Rastrigin has periodic minima. Uniform sampling is good.
        for i in range(explore_budget):
            if check_time():
                break
            
            # Mix of center and random points
            if i == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng.uniform(-5.0, 5.0, size=n)
            
            fx = safe_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
                
            if calls_made >= budget:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x0 = np.zeros(n)
            best_f = safe_func(x0)
            best_x = x0.copy()

        # Phase 2: Local Polishing
        # Use Nelder-Mead or L-BFGS-B from the best points found.
        # Since we don't know the gradient, Nelder-Mead is a good derivative-free choice.
        # However, Nelder-Mead can be slow. 
        # Alternative: Coordinate Descent with adaptive steps is very fast for smooth-ish functions.
        # But Rastrigin is non-smooth. 
        # Let's use a hybrid: Try Nelder-Mead with a tight maxiter, then refine with small coordinate steps.
        
        # Actually, for Rastrigin, the local minima are quite narrow.
        # A more effective local strategy for Rastrigin is often a simple adaptive coordinate search 
        # or a (1+1)-ES with small sigma.
        
        # Let's implement a fast adaptive coordinate descent / hill climbing.
        # This is much faster than Nelder-Mead for high dimensions and many restarts.
        
        # We will run local search from the best point found in Phase 1.
        # We might also run it from the second best if time permits.
        
        # Strategy: 
        # 1. Take the best point.
        # 2. Perform adaptive coordinate descent.
        # 3. If stuck, try small random perturbations (simulated annealing lite).
        
        # Let's stick to a robust Nelder-Mead implementation from scipy for the main local phase,
        # but limit iterations to ensure we finish within time.
        
        # However, scipy.optimize.minimize has overhead. 
        # Given n=5, a custom (1+1)-ES or CMA-ES is better.
        # But the prompt says "DIFFERENT strategy". CMA-ES is the current one.
        # So let's do **Multi-Start Nelder-Mead**.
        
        # To be efficient, we'll do Nelder-Mead on the top few candidates from Phase 1.
        
        # Sort candidates by fitness? We only kept the best. 
        # Let's keep a list of top candidates from Phase 1.
        
        # Re-do Phase 1 to keep top K
        candidates = []
        for i in range(explore_budget):
            if check_time():
                break
            if i == 0:
                x0 = np.zeros(n)
            else:
                x0 = rng.uniform(-5.0, 5.0, size=n)
            fx = safe_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            # Keep top 5 candidates
            candidates.append((fx, x0.copy()))
            if len(candidates) > 5:
                candidates.sort(key=lambda x: x[0])
                candidates = candidates[:5]
            
            if calls_made >= budget:
                break
        
        # Phase 2: Local Optimization on top candidates
        # Use Nelder-Mead with adaptive initial simplex size.
        
        for fx, x0 in candidates:
            if check_time():
                break
            if calls_made >= budget:
                break
            
            # Initial simplex size: proportional to distance to boundary or just a fixed small value
            # Rastrigin period is 2pi. Local minima are within ~0.5 of the integer grid (shifted).
            # So a step size of 0.5-1.0 is reasonable for initial simplex.
            
            options = {
                'maxiter': 200,  # Limit iterations to prevent getting stuck
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            try:
                res = minimize(safe_func, x0, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # If Nelder-Mead didn't improve much, try a simple coordinate polish
            # This helps with the "jagged" nature of Rastrigin near the minimum.
            if not check_time() and calls_made < budget:
                x_cur = res.x.copy()
                f_cur = res.fun
                step = 0.1
                for dim in range(n):
                    for _ in range(10): # 10 steps per dim
                        if check_time() or calls_made >= budget:
                            break
                        x_plus = x_cur.copy()
                        x_plus[dim] += step
                        x_minus = x_cur.copy()
                        x_minus[dim] -= step
                        
                        f_plus = safe_func(x_plus)
                        f_minus = safe_func(x_minus)
                        
                        if f_plus < f_cur:
                            x_cur = x_plus
                            f_cur = f_plus
                        elif f_minus < f_cur:
                            x_cur = x_minus
                            f_cur = f_minus
                        else:
                            # No improvement, shrink step
                            step *= 0.5
                            if step < 1e-4:
                                break
                        if f_cur < best_f:
                            best_f = f_cur
                            best_x = x_cur.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
