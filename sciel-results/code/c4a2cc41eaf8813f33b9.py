import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x)
            # Clip to box to ensure valid input, though task says func clips
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        # 1. Global Search with Dual Annealing
        # Dual annealing is robust for low-dimensional black-box problems.
        # It uses simulated annealing for global exploration and Nelder-Mead for local polishing.
        # maxiter=1 is a bit aggressive, but we have a follow-up.
        # Actually, let's give it a reasonable number of iterations but rely on the callback to stop if budget runs out.
        # However, dual_annealing doesn't have a direct "budget" stop, it has maxiter and maxfev.
        # We can set maxfev to our remaining budget, but it might not use it all if it converges.
        # If it converges early, we use the remaining budget for the random walk.
        
        # Initial guess: Origin is often a good start, or random.
        # Let's try a few initial points if budget allows? 
        # Dual annealing takes a single x0.
        # Let's just start at 0.
        x0 = np.zeros(self.dim)
        
        # Bounds
        bounds = [(-5, 5)] * self.dim
        
        # Run Dual Annealing
        # maxiter: number of iterations. 
        # maxfev: maximum function evaluations.
        # We want to use most of the budget here.
        # Let's reserve 50 calls for the final polish.
        reserved_calls = 50
        max_feasible = max(10, self.budget - reserved_calls)
        
        try:
            result = so.dual_annealing(
                safe_func,
                x0,
                bounds=bounds,
                maxiter=100, # Allow many iterations
                maxfev=max_feasible,
                seed=42
            )
            
            if result.x is not None:
                if self.best_x is None or result.fun < self.best_f:
                    self.best_x = result.x.copy()
                    self.best_f = result.fun
        except Exception:
            # If dual annealing fails, we still have self.best_x/f from safe_func calls
            pass

        # 2. Final Polish: Random Walk
        # This ensures we use the remaining budget to potentially improve AOCC in the later calls.
        # AOCC rewards early progress, but also that the best value doesn't degrade.
        # Since we keep the global best, the "best value after call i" is non-increasing.
        # So simply making more calls that don't find a better solution doesn't hurt the score,
        # but finding a slightly better one late helps the tail of the curve.
        # A random walk around the current best is a simple, safe way to spend remaining budget.
        
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = safe_func(self.best_x)
            
        x = self.best_x.copy()
        f = self.best_f
        
        # Start with a moderate step size
        sigma = 0.5
        
        while self.calls < self.budget and self._check_time():
            # Generate random step
            z = np.random.standard_normal(self.dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            
            if f_new < f:
                x = x_new
                f = f_new
                # Increase step size slightly on success
                sigma *= 1.2
            else:
                # Decrease step size on failure
                sigma *= 0.85
                
            if sigma < 1e-6:
                # If step size is too small, reset to a small value to continue exploring
                sigma = 0.1

        self.best_x = x if f < self.best_f else self.best_x
        self.best_f = f if f < self.best_f else self.best_f

        return self.best_x, self.best_f
