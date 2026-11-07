import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin from 330s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        
        def check_time():
            return time.time() > deadline - 5

        # Helper to evaluate and track calls
        def eval_func(x):
            nonlocal calls_used
            if check_time():
                return float('inf')
            if calls_used >= budget:
                return float('inf')
            calls_used += 1
            return func(x)

        # Phase 1: Diverse Initialization
        # Combine Halton (low discrepancy) with vertices and center
        # to ensure coverage of the entire [-5, 5]^5 box.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_halton = 15
        points = []
        values = []
        
        # 1. Add center
        points.append(np.zeros(dim))
        
        # 2. Add vertices scaled to [-4, 4] to avoid boundary issues but cover corners
        for i in range(2 ** dim):
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 8.0 - 4.0 # Map 0,1 to -4,4
            points.append(v)
            
        # 3. Add Halton points
        for i in range(1, n_halton + 1):
            x = np.array([halton(i, p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
            x = x * 10.0 - 5.0 # Map to [-5, 5]
            points.append(x)

        # Evaluate all initial points
        for x in points:
            if check_time() or calls_used >= budget:
                break
            f = eval_func(x)
            values.append(f)
            
        if not values:
            # Fallback if we couldn't even make one call (unlikely but safe)
            x = np.zeros(dim)
            f = eval_func(x)
            values.append(f)
            points.append(x)

        # Sort points by value
        sorted_indices = np.argsort(values)
        top_k = 5
        top_indices = sorted_indices[:top_k]
        
        best_x = points[top_indices[0]].copy()
        best_f = values[top_indices[0]]
        
        # Phase 2: Local Optimization using Nelder-Mead
        # Nelder-Mead is robust for small dimensions and multimodal landscapes.
        # We apply it to the top few distinct starting points.
        
        tried_keys = set()
        
        for idx in top_indices:
            if check_time() or calls_used >= budget:
                break
                
            x0 = points[idx].copy()
            f0 = values[idx]
            
            # Skip if too close to a previously tried start
            key = tuple(np.round(x0, 1))
            if key in tried_keys:
                continue
            tried_keys.add(key)
            
            # Reserve some budget for the final ES phase
            reserve = 50
            max_fev_allowed = min(100, budget - calls_used - reserve)
            if max_fev_allowed < 10:
                break
                
            try:
                result = minimize(
                    eval_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': max_fev_allowed,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass

        # Phase 3: Polishing with (1+1)-ES
        # After NM, we are likely near a local minimum.
        # (1+1)-ES is very efficient for final convergence and handles the
        # fine structure of Katsuura well.
        
        if calls_used < budget and not check_time() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Start with a small step
            min_sigma = 1e-10
            max_sigma = 1.0
            rng = np.random.default_rng(42)
            
            while calls_used < budget and not check_time():
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase step size on success
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.9, min_sigma)
                    if sigma <= min_sigma * 1.01:
                        break

        return best_x, best_f
