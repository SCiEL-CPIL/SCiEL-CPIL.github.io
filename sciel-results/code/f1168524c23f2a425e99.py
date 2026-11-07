import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        evaluated_points = []
        
        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            val = func(x)
            calls_made += 1
            evaluated_points.append((x, val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Structured random sampling
        # Katsuura has structure: product of cosines + polynomial
        # The polynomial part has minimum near x=1 in each dimension
        # Cosine terms create many local minima
        # Strategy: sample near structured points + random
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # 1. Near the polynomial minimum (x=1)
        # 2. Random points in the box
        # 3. Points near cosine minima (multiples of pi)
        
        n_phase1 = min(50, budget)
        phase1_points = []
        
        # Points near x=1 (polynomial minimum)
        for i in range(10):
            x = np.ones(dim) + rng.normal(0, 0.5, size=dim)
            phase1_points.append(x)
        
        # Random points
        for i in range(20):
            x = rng.uniform(-5, 5, size=dim)
            phase1_points.append(x)
        
        # Points near cosine minima (multiples of pi)
        for i in range(20):
            x = rng.uniform(-5, 5, size=dim)
            # Snap to nearby multiples of pi
            x = np.round(x / np.pi) * np.pi
            # Add small noise
            x = x + rng.normal(0, 0.1, size=dim)
            phase1_points.append(x)
        
        for x in phase1_points[:n_phase1]:
            safe_call(x)
            if calls_made >= budget or time.time() > deadline:
                break
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Multi-start CMA-ES
        # Sort by fitness and run CMA-ES on top candidates
        evaluated_points.sort(key=lambda item: item[1])
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        # CMA-ES in 5D typically needs ~100-200 evaluations per run
        # We'll run on the top few candidates
        top_k = min(5, len(evaluated_points))
        
        for i in range(top_k):
            if calls_made >= budget or time.time() > deadline:
                break
            
            x0 = evaluated_points[i][0].copy()
            
            # Determine maxfev for this CMA-ES run
            # Allocate remaining budget proportionally
            maxfev = min(150, remaining - i * 10)
            if maxfev < 20:
                break
            
            def cma_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return best_f
                return safe_call(x)
            
            try:
                # CMA-ES via scipy
                res = minimize(
                    cma_func,
                    x0,
                    method='CMA',
                    options={
                        'maxiter': 100,
                        'maxfev': maxfev,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'bounds': [(-5, 5)] * dim
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
