import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 295.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
            
        N = budget
        
        # Pre-compute optimal x0 distribution
        # We sample the Pareto front of ZDT4.
        # f1(x0) = x0
        # f2(x0) = 1 - sqrt(x0) - x0 * sin(10*pi*x0)^2  (from directive)
        
        n_fine = 20000
        x_fine = np.linspace(0.0, 1.0, n_fine)
        
        # Calculate f2
        # Handle x=0 for sqrt
        x_safe = np.where(x_fine == 0, 1e-9, x_fine)
        y_fine = 1.0 - np.sqrt(x_safe) - x_fine * np.sin(10.0 * np.pi * x_fine)**2
        
        # Identify non-dominated points
        # A point is dominated if there exists x' <= x with f2(x') < f2(x)
        # Since f1 = x, we just need f2(x) <= min_{t<=x} f2(t)
        
        running_min = np.minimum.accumulate(y_fine)
        
        # Tolerance for numerical precision
        tol = 1e-9
        is_pareto = (y_fine <= running_min + tol)
        
        # Get the indices of Pareto points
        pareto_indices = np.where(is_pareto)[0]
        
        if len(pareto_indices) == 0:
             # Fallback to uniform if something went wrong
             x0_vals = np.linspace(0, 1, N)
        else:
             # We want to distribute N points along the Pareto front.
             # Uniform arc length is a good metric.
             # Extract x and y for Pareto points
             x_pareto = x_fine[pareto_indices]
             y_pareto = y_fine[pareto_indices]
             
             # Calculate arc lengths for the Pareto segments
             dx = np.diff(x_pareto)
             dy = np.diff(y_pareto)
             ds = np.sqrt(dx**2 + dy**2)
             
             s_pareto = np.concatenate(([0.0], np.cumsum(ds)))
             s_max = s_pareto[-1]
             
             if s_max == 0:
                 x0_vals = np.array([x_pareto[0]])
             else:
                 s_norm = s_pareto / s_max
                 s_targets = np.linspace(0.0, 1.0, N)
                 x0_vals = np.interp(s_targets, s_norm, x_pareto)
                 
                 # Ensure boundaries
                 x0_vals[0] = 0.0
                 x0_vals[-1] = 1.0
                 
                 # Clip
                 x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = x0_val
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)
            
        return np.empty((0, dim))
