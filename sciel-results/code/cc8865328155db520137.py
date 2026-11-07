import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 280  # 280s safety margin out of 300s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f
        
        # Use L-BFGS-B with finite differences
        # This is a structural change from Nelder-Mead to a gradient-based quasi-Newton method.
        # L-BFGS-B is much more efficient for Rosenbrock's narrow valley.
        
        def local_optimize(x0, f0, max_iter):
            nonlocal best_f, best_x, calls
            
            # We need to pass the function to minimize.
            # minimize will call func_wrapper which calls eval_func.
            # We need to ensure we don't exceed budget.
            
            def func_wrapper(x):
                if calls >= budget:
                    return best_f
                return eval_func(x)
            
            # Bounds are [-5, 5]
            bounds = [(-5, 5)] * dim
            
            try:
                res = minimize(func_wrapper, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                
                # Check if the result is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            return res.x, res.fun if hasattr(res, 'fun') else best_f

        # Multi-start strategy
        # With L-BFGS-B, we can afford more restarts or longer iterations per restart.
        # Let's try 10 restarts.
        n_starts = 10
        # L-BFGS-B uses multiple calls per iteration (finite diff gradient).
        # For dim=5, gradient takes 6 calls (central) or 5 (forward).
        # Let's allocate a moderate max_iter to allow convergence but leave room for restarts.
        # If each restart takes ~50-100 calls, 10 restarts = 500-1000 calls.
        # We have 500 calls. So maybe 5 restarts with max_iter=50?
        # Or 10 restarts with max_iter=20?
        # Rosenbrock in 5D might need more than 20 iterations to converge tightly if starting far.
        # Let's try 5 restarts with max_iter=80.
        
        n_starts = 5
        max_iter_per_start = 80
        
        for s in range(n_starts):
            if time.time() > self.deadline:
                break
            if calls >= budget:
                break
            
            if s == 0:
                x0 = np.zeros(dim)
            else:
                # Sample from a wider range to find the basin
                x0 = rng.uniform(-4.5, 4.5, dim)
            
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
            
            # Run L-BFGS-B
            x_opt, f_opt = local_optimize(x0, f0, max_iter_per_start)
        
        # Final polish if budget remains
        if best_x is not None and time.time() < self.deadline and calls < budget:
            # Use the remaining budget for a final high-precision L-BFGS-B run
            remaining_budget = budget - calls
            # Estimate remaining iterations. Each iteration ~ 6 calls.
            max_iter_polish = max(1, remaining_budget // 10)
            local_optimize(best_x, best_f, max_iter_polish)

        return best_x, best_f
