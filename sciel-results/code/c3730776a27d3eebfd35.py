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
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # --- Phase 1: Global Search ---
        # BBOB-21 (Gallagher 101 peaks) has many local optima.
        # We need a good initial spread to find the basin of the global optimum.
        # Use 30% of the budget for global search to allow more time for local refinement.
        
        n_global = int(budget * 0.3)
        n_global = max(n_global, 30)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            try:
                f_val = func(lhs[i])
                f_values[i] = f_val
                calls += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = lhs[i].copy()
            except Exception:
                break

        if calls == 0:
            return best_x, 0.0

        # Sort by fitness to get top candidates
        sorted_indices = np.argsort(f_values)
        
        # --- Phase 2: Local Refinement ---
        # Use Powell method for local refinement, which is robust and efficient in low dimensions.
        # We will try the top 5 candidates from the global search.
        
        n_local_candidates = 5
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        budget_per_local = max(10, remaining_budget // n_local_candidates)
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = lhs[sorted_indices[k]].copy()
            f0 = f_values[sorted_indices[k]]
            
            # Wrapper to count calls and enforce budget
            local_calls = 0
            max_local_calls = budget_per_local
            
            def local_func(x):
                nonlocal calls, local_calls, best_f, best_x
                if calls >= budget or local_calls >= max_local_calls or time.time() > deadline:
                    raise TimeoutError("Budget exceeded")
                calls += 1
                local_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                # Use Powell method
                res = minimize(local_func, x0, method='Powell', options={'maxiter': 100})
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
