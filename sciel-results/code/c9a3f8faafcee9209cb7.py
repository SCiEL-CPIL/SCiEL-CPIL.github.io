import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        d = self.dim
        rng = np.random.default_rng(42)
        
        # State tracking
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def time_ok():
            return (time.time() - start_time) < timeout

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Differential Evolution for Global Search
        # Allocate 60% of the budget to DE
        de_budget = int(self.budget * 0.6)
        popsize = 15
        maxiter = max(1, de_budget // popsize)
        
        bounds = [(-5.0, 5.0)] * d
        
        try:
            res_de = scipy.optimize.differential_evolution(
                safe_func,
                bounds,
                popsize=popsize,
                maxiter=maxiter,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass

        # Fallback: If no calls were made or best_x is None, evaluate center
        if best_x is None:
            x_center = np.zeros(d)
            try:
                fx = safe_func(x_center)
                best_x = x_center.copy()
                best_f = fx
            except Exception:
                pass

        # Phase 2: Local Random Perturbation Search
        # This helps cross the step boundary if DE stopped on the plateau
        if best_x is not None and calls_made < self.budget and time_ok():
            radii = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            current_best_x = best_x.copy()
            current_best_f = best_f
            
            # Distribute remaining budget (approx 40% = 200 calls) among radii
            # Let's use 15 points per radius, 7 radii = 105 calls
            points_per_radius = 15
            
            for r in radii:
                if not time_ok() or calls_made >= self.budget:
                    break
                
                for _ in range(points_per_radius):
                    if not time_ok() or calls_made >= self.budget:
                        break
                    
                    # Random direction
                    dir = rng.standard_normal(d)
                    norm = np.linalg.norm(dir)
                    if norm > 0:
                        dir /= norm
                    
                    # Random distance in [0, r]
                    dist = rng.uniform(0, r)
                    
                    x_candidate = current_best_x + dist * dir
                    x_candidate = np.clip(x_candidate, -5.0, 5.0)
                    
                    fx = safe_func(x_candidate)
                    
                    if fx < current_best_f:
                        current_best_f = fx
                        current_best_x = x_candidate.copy()
                        update_best(current_best_x, current_best_f)

        # Phase 3: Local Refinement with Powell's Method
        # Powell is good for the smooth ellipsoidal part inside the step
        if best_x is not None and calls_made < self.budget and time_ok():
            remaining_budget = self.budget - calls_made
            # Powell's method call count is roughly d * maxiter
            maxiter_powell = max(1, remaining_budget // (d + 1))
            
            try:
                res_powell = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={
                        'xtol': 1e-8,
                        'ftol': 1e-10,
                        'maxiter': maxiter_powell,
                        'disp': False
                    }
                )
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = res_powell.x.copy()
            except Exception:
                pass

        return best_x, best_f
