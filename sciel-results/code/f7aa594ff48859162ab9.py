import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            # func clips to box, but we ensure x is valid
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
            return val

        dim = self.dim
        budget = self.budget
        bounds = [(-5, 5)] * dim

        # Phase 1: Global Search with Differential Evolution
        # Use a portion of the budget for global exploration
        # DE uses popsize * dim evaluations per generation roughly, but actually popsize evaluations per generation.
        # Let's allocate 60% of the budget to DE.
        de_budget = int(budget * 0.6)
        
        # We need to map DE's internal calls to our safe_call.
        # differential_evolution does not allow a custom call counter directly in older scipy, 
        # but we can wrap the function.
        
        # To control the number of calls strictly, we might need to implement a simple DE 
        # or use a wrapper that stops if budget is exceeded.
        # However, scipy's DE is robust. Let's try to constrain it.
        
        # A safer approach: Implement a simple multi-start local search.
        # Given the budget is small (500), a well-tuned CMA-ES or Multi-start NM is better.
        
        # Let's try Multi-start Nelder-Mead.
        # Number of starts: 10
        # Budget per start: (budget - 1) / 10 = 50
        # This might be too small for NM to converge.
        
        # Alternative: One global search + one local refinement.
        
        # Let's use a custom simple DE implementation to strictly control calls.
        # Or just use scipy DE with a wrapper that raises an exception if budget is hit.
        
        def wrapped_func(x):
            return safe_call(x)
            
        try:
            # Use differential_evolution
            # popsize=5 is small, might get stuck. popsize=10 is better.
            # maxiter is the number of generations.
            # We don't know exactly how many calls DE will make, so we rely on the wrapper to stop.
            # But if it raises an exception, we lose the current best? No, safe_call updates best_f.
            
            # To be safe, let's do a manual loop for a simple evolutionary strategy or use
            # scipy minimize with Nelder-Mead from multiple random starts.
            
            # Let's do 5 random starts with Nelder-Mead.
            # Budget per start: (budget - 5) // 5 = 99
            n_starts = 5
            budget_per_start = (budget - n_starts) // n_starts
            
            rng = np.random.default_rng(42)
            
            for i in range(n_starts):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Random start point
                x0 = rng.uniform(-5, 5, dim)
                f0 = safe_call(x0)
                
                # Use Nelder-Mead
                # maxiter is not strictly function calls, but iterations.
                # Each iteration in NM uses dim+1 function calls.
                # So maxiter ~ budget_per_start / (dim + 1)
                max_iter = max(1, budget_per_start // (dim + 1))
                
                try:
                    res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        except Exception:
            pass

        return best_x, best_f
