import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.calls = 0
        self.deadline = time.time() + 290

    def __call__(self, problem, bounds):
        self.calls = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        rng = np.random.default_rng(42)
        
        # Generate weight vectors
        # For 3 objectives, use a grid
        # Or random
        n_weights = 10
        weights = []
        for _ in range(n_weights):
            w = rng.random(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
            
        # Initial points: random
        # We will use the best point found so far as the start for local search?
        # Or just start each local search from a random point?
        # Better: Start from a random point, run local search.
        
        # We need to distribute the budget.
        # 200 budget.
        # 10 weights.
        # 20 evals per weight.
        
        # Wrapper for problem to count calls and check deadline
        def wrapped_problem(x):
            if self.calls >= self.budget or time.time() > self.deadline:
                raise ValueError("Budget or Time Exceeded")
            self.calls += 1
            # Clip x to bounds
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            return f
            
        # We need a scalarized objective for each weight
        # Weighted Sum: sum(w_i * f_i)
        # This works well if the Pareto front is convex. WFG9 is likely non-convex.
        # Tchebycheff is better for non-convex.
        
        # Let's use Tchebycheff scalarization for the local search too.
        # But we need the ideal point.
        # We can estimate the ideal point from the first few random evaluations.
        
        # Phase 1: Random Sampling to estimate Ideal Point
        n_init = 10
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([wrapped_problem(x) for x in X_init])
        ideal = np.min(F_init, axis=0)
        
        # Phase 2: Local Search for each weight
        for w in weights:
            if self.calls >= self.budget:
                break
                
            # Start point: random
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            def obj_func(x):
                f = wrapped_problem(x)
                # Tchebycheff
                # g = max(w_i * |f_i - ideal_i|)
                diff = np.abs(f - ideal)
                return np.max(diff * w)
                
            # Run Powell
            # Limit function calls?
            # Powell doesn't have a direct maxfev, but we can use a wrapper.
            # Or use Nelder-Mead which has maxfev.
            # Nelder-Mead is better for controlling budget.
            
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'maxfev': 20, 'xatol': 1e-4, 'fatol': 1e-6})
            except ValueError:
                break
            except Exception:
                break
                
        return None
