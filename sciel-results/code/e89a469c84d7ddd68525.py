import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim
            self.call_count = 0
            self.start_time = time.time()
            self.time_limit = 320.0 # Safety margin

        def __call__(self, func):
            self.call_count = 0
            self.start_time = time.time()
            
            def wrapped_func(x):
                self.call_count += 1
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError
                x = np.clip(x, -5.0, 5.0)
                return func(x)

            n = self.dim
            budget = self.budget
            bounds = [(-5.0, 5.0) for _ in range(n)]
            
            # Phase 1: Random Sampling
            n_starts = 10
            rng = np.random.default_rng(42)
            X0 = rng.uniform(-5.0, 5.0, size=(n_starts, n))
            
            best_x = None
            best_f = float('inf')
            
            # Evaluate initial points
            for x in X0:
                if self.call_count >= budget:
                    break
                f = wrapped_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Phase 2: Local Optimization
            # Sort initial points by fitness to prioritize good starts
            # We need to keep track of which x gave which f
            initial_results = []
            # Re-evaluating is bad. We should have stored them.
            # Let's re-do Phase 1 storing results.
            
            # Reset for clean logic
            self.call_count = 0
            best_x = None
            best_f = float('inf')
            candidates = []
            
            for x in X0:
                if self.call_count >= budget:
                    break
                f = wrapped_func(x)
                candidates.append((f, x.copy()))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Sort candidates
            candidates.sort(key=lambda t: t[0])
            
            # Try to improve the best candidates
            # Use L-BFGS-B
            # Budget remaining: budget - self.call_count
            remaining = budget - self.call_count
            
            # Allocate budget to top 3 candidates
            # Each L-BFGS-B run gets ~ remaining/3 calls
            calls_per_opt = max(10, remaining // 3)
            
            for i, (f0, x0) in enumerate(candidates[:3]):
                if self.call_count >= budget:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break
                
                try:
                    res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxfun': calls_per_opt, 'maxiter': 50})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            # Phase 3: Final Polish
            # If we have budget left, do one more polish on the absolute best
            if self.call_count < budget - 10 and best_x is not None:
                try:
                    res = minimize(wrapped_func, best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxfun': budget - self.call_count - 5, 'maxiter': 100})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            return best_x, best_f
