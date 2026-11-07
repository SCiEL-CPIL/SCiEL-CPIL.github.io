import numpy as np
import scipy.optimize as opt
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            return func(np.clip(x, -5, 5))
        
        # Phase 1: Global Exploration with Sobol Sequence
        n_initial = 50
        n_initial = min(n_initial, max_calls // 2)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate points in [0, 1]^d
        sobol_points = sampler.random(n_initial)
        # Scale to [-5, 5]^d
        initial_points = sobol_points * 10 - 5
        
        initial_values = []
        
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                initial_values.append((fp, p))
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Multi-Start Local Optimization
        initial_values.sort(key=lambda x: x[0])
        
        remaining_budget = max_calls - used_calls
        
        if remaining_budget > 10:
            # Determine number of restarts
            # We have 50 initial points. Let's take the best 5.
            # If we have more budget, we can take more.
            # 500 - 50 = 450 remaining.
            # 5 restarts * 90 calls = 450.
            max_restarts = 5
            
            # Adjust if budget is tight
            budget_per_restart = remaining_budget // max_restarts
            if budget_per_restart < 20:
                max_restarts = max(1, remaining_budget // 20)
                budget_per_restart = remaining_budget // max_restarts
            
            candidates = initial_values[:max_restarts]
            
            for fp, x_start in candidates:
                if used_calls >= max_calls or time.time() > deadline:
                    break
                
                local_budget = max_calls - used_calls
                if local_budget < 10:
                    break
                
                try:
                    result = opt.minimize(
                        safe_func, 
                        x_start, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget, 
                            'maxfev': local_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except (RuntimeError, Exception):
                    if time.time() > deadline or used_calls >= max_calls:
                        break
                    continue
        
        # Phase 3: Final Polish
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 10:
                try:
                    # Use Powell for final polish? It's often better for smooth parts.
                    # But Katsuura is not smooth. NM is safer.
                    result = opt.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (RuntimeError, Exception):
                    pass

        return best_x, best_f
