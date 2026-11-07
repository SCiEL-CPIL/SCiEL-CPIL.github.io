import numpy as np
import time
import scipy.optimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 285  # Safety margin out of 300s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Initial sampling
        n_starts = 10
        candidates = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            candidates.append(rng.uniform(-4, 4, dim))
        
        for x0 in candidates:
            if calls >= budget or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        # Phase 2: L-BFGS-B
        # We run L-BFGS-B from the current best point.
        # We allow it to use the remaining budget.
        while calls < budget and time.time() < self.deadline:
            x_start = best_x.copy()
            
            def scipy_func(x):
                return eval_func(x)
                
            remaining = budget - calls
            if remaining < 10:
                break
                
            try:
                res = scipy.optimize.minimize(
                    scipy_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                break
            
            # Update best if scipy found something better (eval_func already does this, 
            # but res.fun might be slightly different if the last call was not the best? 
            # No, eval_func tracks the global best. res.fun is the final point's value.
            # We trust best_f tracked in eval_func.)
            
            # If L-BFGS-B converged, we are likely done.
            if res.success:
                break
            
            # If it didn't converge, it likely hit maxfun or maxiter.
            # If it hit maxfun, calls == budget (approx), so loop will end.
            # If it hit maxiter, we might have budget left.
            # In that case, we can restart from the new best point.
            # However, L-BFGS-B usually converges or hits maxfun.
            # To be safe, if we have significant budget left and it didn't succeed, 
            # we might try a new random start? 
            # For Rosenbrock, restarting from the current best is usually better than a random jump.
            # But if we are stuck in a local min (unlikely), a jump might help.
            # Given the unimodal nature, let's just break if we have used > 80% of budget 
            # or if we made little progress.
            
            # Simple check: if we have more than 20 calls left, and it didn't succeed, 
            # we try again from the best point.
            if budget - calls > 20:
                continue
            else:
                break

        return best_x, best_f
