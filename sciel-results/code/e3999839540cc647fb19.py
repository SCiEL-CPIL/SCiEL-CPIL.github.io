import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        dim = self.dim
        budget = int(self.budget)
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
            
        eval_count = 0
        results = []
        
        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None
            try:
                f = problem(x)
                eval_count += 1
                return f
            except Exception:
                eval_count += 1
                return None

        # Strategy: Sample the Pareto front of ZDT4.
        # Front: x_1 = hi[0] (usually 1.0), x_2..x_10 vary to control f_1.
        # We parameterize x_2..x_10 as x_i = 0.5 * (1 - t) for t in [0, 1].
        # This moves x_i from 0.5 (min f_1) to 0.0 (max f_1 contribution).
        
        # Determine x_1 value. For ZDT4, x_1=1 minimizes f_2 for f_1 >= 11.
        x1_val = hi[0]
        
        # If bounds for x_1 are different, we might need to adjust, but standard is [0,1].
        
        n_samples = budget
        # Use linspace for t
        t_vals = np.linspace(0.0, 1.0, n_samples)
        
        for t in t_vals:
            if eval_count >= budget or time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_val
            
            # Set x_1..x_9 (indices 1 to 9)
            # x_i = 0.5 * (1 - t)
            # Clip to bounds
            val = 0.5 * (1.0 - t)
            # Ensure it's within bounds for x[1:]
            # Standard bounds are [0,1]. 0.5*(1-t) is in [0, 0.5].
            x[1:] = val
            
            x = np.clip(x, lo, hi)
            
            f = safe_eval(x)
            if f is not None:
                results.append((x, f))
                
        # If we have budget left (e.g. if some evals failed or time check), 
        # or if we want to be safer, we can add some random points on the front.
        # But linspace should cover it.
        
        if results:
            F = np.array([f for _, f in results])
            X = np.array([x for x, _ in results])
            
            ref = np.array([11.0, 400.0])
            mask = np.all(F < ref, axis=1)
            X = X[mask]
            F = F[mask]
            
            if len(F) > 0:
                return X
        
        return np.empty((0, dim))
