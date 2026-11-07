import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Optimizer using Random Sampling followed by multiple local searches.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard variables
        start_time = time.time()
        deadline = 310.0  # Safety margin from 330s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            # Check deadline
            if time.time() > deadline:
                return float('inf')
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Random Exploration
        # Use a fraction of the budget for exploration
        # 30 random points is a good balance for dim=5
        n_explore = min(30, self.budget // 10)
        
        for _ in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # Phase 2: Local Exploitation using multiple local searches
        # We'll do a few local searches from the best points found
        # First, let's collect the top few points from exploration
        
        # Actually, let's just do Powell from the best point, and maybe a few others
        if best_x is not None and calls < self.budget:
            remaining = self.budget - calls
            # Split remaining budget into chunks for multiple local searches
            # Let's do 2 local searches: one from best, one from a random point
            n_local = 2
            budget_per_local = remaining // n_local
            
            for i in range(n_local):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                if i == 0:
                    start_x = best_x.copy()
                else:
                    # Start from a random point, but biased towards the best region
                    start_x = rng.uniform(-5, 5, self.dim)
                    # Or we could perturb the best point
                    # start_x = best_x + 0.5 * rng.standard_normal(self.dim)
                    # start_x = np.clip(start_x, -5, 5)
                
                try:
                    res = minimize(
                        wrapped_func, 
                        start_x, 
                        method='Powell', 
                        options={
                            'maxfev': budget_per_local,
                            'xtol': 1e-8,
                            'ftol': 1e-8
                        }
                    )
                except Exception:
                    pass

        # If we still have budget, do another local search from best
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={
                        'maxfev': remaining,
                        'xtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass

        return best_x, best_f
