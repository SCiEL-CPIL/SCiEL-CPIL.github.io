"""Optimizer for BBOB f19 (Griewank-Rosenbrock): ILS with Pattern Search."""
import numpy as np
import time
import math

class Optimizer:
    """ILS with Pattern Search for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        def pattern_search(x0, max_calls, step_size=0.5):
            """Pattern Search local optimization."""
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            current_x = x0.copy()
            current_f = safe_eval(current_x)
            if current_f is None:
                return
            
            # Directions: ±e_i for each dimension
            directions = []
            for i in range(dim):
                d_pos = np.zeros(dim)
                d_pos[i] = 1.0
                directions.append(d_pos)
                directions.append(-d_pos)
            
            # Pattern search loop
            improved = True
            while improved and calls_made < budget and time.time() < deadline:
                improved = False
                
                # Try all directions
                for d in directions:
                    if calls_made >= budget or time.time() >= deadline:
                        return
                    
                    x_try = current_x + step_size * d
                    x_try = np.clip(x_try, -5, 5)
                    f_try = safe_eval(x_try)
                    if f_try is None:
                        return
                    
                    if f_try < current_f:
                        current_x = x_try
                        current_f = f_try
                        improved = True
                        # Expand step size if we found improvement
                        step_size = min(step_size * 1.5, 2.0)
                        break
                else:
                    # No improvement in any direction, shrink step size
                    step_size *= 0.5
                    if step_size < 0.01:
                        break

            return

        # Phase 1: Initial diverse sampling
        n_init = 15
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for x0 in init_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x0)
            if fx is None:
                break

        # Phase 2: Iterated Local Search
        # Alternate between local refinement and perturbation
        while calls_made < budget and time.time() < deadline:
            # Local polish from current best
            if calls_made < budget and time.time() < deadline:
                remaining = budget - calls_made
                if remaining > 10:
                    ps_budget = min(remaining, 40)
                    pattern_search(best_x, ps_budget, step_size=0.3)
            
            # Perturb best to escape local minima
            if calls_made < budget and time.time() < deadline:
                # Adaptive perturbation size
                perturbation_size = 0.5 + 0.5 * (calls_made / budget)
                x_perturbed = best_x + rng.standard_normal(dim) * perturbation_size
                x_perturbed = np.clip(x_perturbed, -5, 5)
                fx = safe_eval(x_perturbed)
                if fx is None:
                    break

        # Final polish with remaining budget
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 5:
                pattern_search(best_x, remaining, step_size=0.1)

        return best_x, best_f
