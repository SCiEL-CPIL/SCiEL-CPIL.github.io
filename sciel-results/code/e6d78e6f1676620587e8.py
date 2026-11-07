import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start L-BFGS-B Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    - Phase 1: Generate diverse starting points using Halton sequence + random.
    - Phase 2: Run L-BFGS-B from each start point, sharing the global call budget.
    - Phase 3: Polish the best solution with tighter tolerances if budget remains.
    - L-BFGS-B is highly efficient for smooth functions like Rosenbrock, using
      quasi-Newton approximations to converge rapidly along the valley.
    - We use finite differences for gradients since the function is a black-box.
    - We carefully manage the call budget to ensure we do not exceed 500 calls.
    - We track the best solution found across all local optimizations.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)
        
    def _halton(self, n, dim, skip=1):
        """Generate Halton sequence points in [0,1]^dim."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for base in range(2, dim + 2):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    i_val, rem = divmod(i_val, base)
                    f /= base
                    r += f * rem
                point.append(r)
            points.append(point)
        return np.array(points)
    
    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # Safety margin for 330s limit
        
        n = self.dim
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            calls_made += 1
            return func(x_clipped)
        
        # Generate diverse starting points using Halton + random
        n_starts = 10
        starts = []
        
        # Include origin
        starts.append(np.zeros(n))
        
        # Halton sequence mapped to [-4, 4]
        halton_points = self._halton(n_starts - 2, n)
        for hp in halton_points:
            x_start = hp * 8 - 4  # Map [0,1] to [-4,4]
            starts.append(x_start)
        
        # Add a few random points for extra diversity
        for _ in range(2):
            x_start = self.rng.uniform(-4, 4, size=n)
            starts.append(x_start)
        
        # For each start, run L-BFGS-B
        for x0 in starts:
            if calls_made >= self.budget - 20:  # Reserve budget for polishing
                break
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    tracked_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                continue
                
        # Polish the best solution with tighter settings if budget remains
        if best_x is not None and calls_made < self.budget - 5 and time.time() < deadline - 5:
            try:
                res = minimize(
                    tracked_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # If no valid result, fallback to origin
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
