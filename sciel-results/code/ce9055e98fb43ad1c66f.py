import numpy as np
import time
import scipy.optimize
import scipy.stats

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x = np.clip(x, -5, 5)
            v = func(x)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # Phase 1: Global Initialization with Latin Hypercube Sampling
        # We use a subset of the budget for initialization
        n_init = 30
        if n_init > self.budget:
            n_init = self.budget

        # Generate LHS points
        try:
            sampler = scipy.stats.qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=n_init)
            # Scale from [0, 1] to [-5, 5]
            points = points * 10.0 - 5.0
        except Exception:
            points = rng.uniform(-5, 5, (n_init, dim))

        # Evaluate initial points
        init_values = np.array([wrapped_func(p) for p in points])
        
        # Phase 2: Select best candidates for local refinement
        # We pick the top 5 unique best points
        best_indices = np.argsort(init_values)[:5]
        candidates = [points[i] for i in best_indices]
        
        # Phase 3: Local Refinement using Nelder-Mead
        # We allocate a portion of the remaining budget for each candidate
        remaining_budget = self.budget - calls
        max_iter_per_candidate = 20
        
        for cand in candidates:
            if calls >= self.budget - 10:
                break
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    cand, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_per_candidate, 'xatol': 1e-8, 'fatol': 1e-12}
                )
                # Update best if needed (wrapped_func does this automatically)
            except Exception:
                pass

        # Phase 4: Iterative Refinement around the global best
        # If we still have budget, we try to improve the global best further
        if best_x is not None:
            step_size = 1.0
            while calls < self.budget - 20 and time.time() - start_time < time_limit - 5.0:
                # Generate a few points around the current best
                n_new = 5
                new_points = [best_x + rng.normal(0, step_size, dim) for _ in range(n_new)]
                new_values = [wrapped_func(p) for p in new_points]
                
                # Find the best new point
                best_new_idx = np.argmin(new_values)
                best_new_x = new_points[best_new_idx]
                best_new_f = new_values[best_new_idx]
                
                if best_new_f < best_f:
                    best_x = best_new_x
                    best_f = best_new_f
                    # Shrink step size slightly as we get closer
                    step_size *= 0.5
                else:
                    # If no improvement, shrink step size or break
                    step_size *= 0.5
                    if step_size < 0.01:
                        break

                # Try to polish the new best with a few Nelder-Mead iterations
                if calls < self.budget - 5:
                    try:
                        res = scipy.optimize.minimize(
                            wrapped_func, 
                            best_x, 
                            method='Nelder-Mead',
                            options={'maxiter': 10, 'xatol': 1e-8, 'fatol': 1e-12}
                        )
                    except Exception:
                        pass

        return best_x, best_f
