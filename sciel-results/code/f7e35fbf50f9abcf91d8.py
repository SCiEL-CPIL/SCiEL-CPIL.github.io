import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        N = self.budget
        
        def time_ok():
            return (time.time() - start_time) < deadline

        rng = np.random.default_rng(0)
        evals = 0
        
        def run_es(x_start, fx_start, budget, sigma_init):
            nonlocal evals
            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            
            for _ in range(budget):
                if evals >= N or not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                evals += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
            return x, fx

        # Phase 1: Multi-start global search
        restarts = 10
        budget_per_restart = max(5, (N // 2) // restarts)
        
        best_x = None
        best_f = np.inf
        
        for i in range(restarts):
            if evals >= N or not time_ok():
                break
            x0 = rng.uniform(-5, 5, d)
            f0 = func(x0)
            evals += 1
            x, f = run_es(x0, f0, budget_per_restart, 2.0)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        # Phase 2: Local refinement
        if best_x is not None:
            remaining = N - evals
            if remaining > 0 and time_ok():
                # Refine from best found
                x, f = run_es(best_x, best_f, remaining, 0.5)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
        # Phase 3: Final Polish (optional, if time left)
        if best_x is not None and time_ok() and evals < N:
             # Use a very small step size for final convergence
             # We don't know how many evals are left exactly, but run_es handles it
             # Let's just run one more short pass with small sigma
             x, f = run_es(best_x, best_f, min(20, N - evals), 0.1)
             if f < best_f:
                 best_f = f
                 best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            
        return best_x, best_f
