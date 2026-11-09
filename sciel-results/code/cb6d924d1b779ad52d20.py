import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 60% of budget for global exploration
        global_budget = int(self.budget * 0.6)
        
        # (1+1)-ES with adaptive sigma
        x = rng.uniform(-5, 5, self.dim)
        sigma = 1.0
        stall_count = 0
        stall_limit = 50
        best_local = best_f
        best_local_x = best_x.copy() if best_x is not None else x.copy()
        
        while calls < global_budget and time.time() < deadline:
            # Generate offspring
            mutation = rng.normal(0, sigma, self.dim)
            x_new = np.clip(x + mutation, -5, 5)
            
            try:
                f_new = safe_func(x_new)
            except (TimeoutError, StopIteration):
                break
            
            if f_new < best_f:
                # Success: accept, shrink sigma
                x = x_new.copy()
                best_f = f_new
                best_x = x_new.copy()
                best_local = f_new
                best_local_x = x_new.copy()
                stall_count = 0
                sigma *= 0.9
            else:
                # Failure: keep x, grow sigma, check for stall
                stall_count += 1
                sigma *= 1.2
                
                if stall_count >= stall_limit:
                    # Restart from a new random point
                    x = rng.uniform(-5, 5, self.dim)
                    sigma = 1.0
                    stall_count = 0
                    try:
                        safe_func(x)
                    except (TimeoutError, StopIteration):
                        break
        
        # Phase 2: Local refinement with Nelder-Mead on the best point found
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls
            if remaining > 10:
                try:
                    from scipy.optimize import minimize
                    result = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining // 2, 200),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        return best_x, best_f
