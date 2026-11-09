import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if not time_ok():
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(v):
                v = best_f
                return v
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization with Latin Hypercube + Origin
        init_points = []
        init_points.append(np.zeros(dim))
        
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=20)
            # Scale from [0,1] to [-4,4]
            points = points * 8.0 - 4.0
            init_points.extend(points)
        except ImportError:
            init_points.extend(rng.uniform(-4, 4, (20, dim)))

        # Evaluate init
        init_values = []
        for p in init_points:
            if not time_ok() or calls >= budget - 100:
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        sorted_idx = np.argsort(init_values)
        # Get top 5 starting points for multi-start
        top_starts = []
        top_starts_f = []
        for idx in sorted_idx[:5]:
            top_starts.append(init_points[idx].copy())
            top_starts_f.append(init_values[idx])

        # 2. Multi-Start (1+1)-ES with Adaptive Sigma
        # We use the top 5 starts.
        # Budget allocation: 150, 100, 80, 60, 40 (Total 430)
        # Init used ~21 calls. Remaining ~479.
        # Total ES budget 430 leaves ~49 for polish.
        step_budgets = [150, 100, 80, 60, 40]
        
        for i, x0 in enumerate(top_starts):
            if not time_ok() or calls >= budget - 50:
                break
            
            x = x0.copy()
            f_x = top_starts_f[i]
            
            sigma = 0.5
            no_improve = 0
            max_steps = step_budgets[i]
            
            for _ in range(max_steps):
                if not time_ok() or calls >= budget - 50:
                    break
                if sigma < 1e-8:
                    break
                    
                # Generate mutant
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                
                f_y = wrapped_func(y)
                
                if f_y < f_x - 1e-10: # Strict improvement
                    x = y
                    f_x = f_y
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if no_improve > 50:
                        break
        
        # 3. Final Polish with Nelder-Mead
        if best_x is not None and calls < budget - 5 and time_ok():
            # Run a tight Nelder-Mead
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        return best_x, best_f
