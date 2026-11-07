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
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Dense grid of starting points for Nelder-Mead
        # Use 25 diverse starting points, each gets ~20 evals for initial NM
        # This is 25 * 20 = 500 evals, but we'll cap by budget
        
        # Generate 25 starting points: mix of structured and random
        start_points = []
        
        # Include origin
        start_points.append(np.zeros(dim))
        
        # Include corners of [-4, 4]^5 (subset)
        corners = []
        for i in range(16):
            c = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            c = c * 8 - 4  # map {0,1} to {-4,4}
            corners.append(c)
        start_points.extend(corners[:15])  # take 15 corners
        
        # Add random points
        while len(start_points) < 25:
            p = rng.uniform(-4.5, 4.5, dim)
            start_points.append(p)
        
        # Phase 2: Run Nelder-Mead from each starting point with limited budget
        nm_options = {
            'maxiter': 15,
            'xatol': 1e-5,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        for sp in start_points:
            if not time_check():
                break
            remaining = budget - calls_made
            if remaining < 15:
                break
            
            try:
                minimize(
                    lambda x: eval_func(x),
                    sp,
                    method='Nelder-Mead',
                    options=nm_options
                )
            except Exception:
                pass
        
        # Phase 3: Refine the best point found with a more aggressive NM
        if time_check() and best_f < np.inf:
            remaining = budget - calls_made
            if remaining > 20:
                nm_refine_options = {
                    'maxiter': min(50, remaining // 6),
                    'xatol': 1e-7,
                    'fatol': 1e-8,
                    'adaptive': True
                }
                try:
                    minimize(
                        lambda x: eval_func(x),
                        best_x,
                        method='Nelder-Mead',
                        options=nm_refine_options
                    )
                except Exception:
                    pass
        
        # Phase 4: Final fine-tuning with (1+1)-ES
        if time_check() and best_f < np.inf:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            
            max_steps = min(50, budget - calls_made)
            for _ in range(max_steps):
                if not time_check():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
