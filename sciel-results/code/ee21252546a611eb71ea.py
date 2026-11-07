import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(self.dim)
        
        # Helper to track calls and deadline
        def eval_func(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError
            if calls_made >= self.budget:
                raise BudgetExceeded
            calls_made += 1
            return func(x)

        class BudgetExceeded(Exception):
            pass

        # 1. Generate Grid Points
        # Step 4.0 in [-5, 5] -> -4, 0, 4
        coords = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*[coords]*self.dim)).T.reshape(-1, self.dim)
        
        # Shuffle grid points
        rng = np.random.default_rng(42)
        rng.shuffle(grid_points)
        
        # Phase 1: Grid Scan
        # Evaluate first 200 points
        grid_limit = 200
        candidates = [] # List of (f, x)
        
        for i in range(min(grid_limit, len(grid_points))):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = grid_points[i]
            try:
                f = eval_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                candidates.append((f, x.copy()))
            except (BudgetExceeded, TimeoutError):
                break
                
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Refine top candidates
        # Select top 5 candidates
        top_k = 5
        top_candidates = candidates[:top_k]
        
        # Allocate remaining budget to top candidates
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Divide budget among top_k
        # Reserve some for safety
        per_candidate_budget = max(10, remaining_budget // top_k)
        
        for f_init, x_init in top_candidates:
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Run Nelder-Mead
            # We need a wrapper that stops when local budget is exceeded
            local_calls = 0
            local_limit = per_candidate_budget
            
            def nm_func(x):
                nonlocal local_calls
                if time.time() > deadline:
                    raise TimeoutError
                if local_calls >= local_limit:
                    raise BudgetExceeded
                if calls_made >= self.budget:
                    raise BudgetExceeded
                local_calls += 1
                calls_made += 1
                return func(x)
                
            try:
                res = minimize(nm_func, x_init, method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (BudgetExceeded, TimeoutError):
                pass
            except Exception:
                # Catch any other errors to keep running
                pass

        return best_x, best_f
