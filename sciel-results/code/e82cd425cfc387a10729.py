import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.func_wrapper = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        # Wrap the function to count calls and update best
        def wrapped_func(x):
            self.calls += 1
            # Clip to bounds to be safe, though L-BFGS-B handles bounds
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        self.func_wrapper = wrapped_func

        # Strategy: Multi-restart L-BFGS-B
        # We will try to launch L-BFGS-B from several diverse starting points.
        # Each run is limited in function calls to ensure we try multiple restarts.
        
        # Generate initial points
        # 1. Center
        # 2. Corners (2^5 = 32 points, too many if each takes 50 calls -> 1600 calls)
        # Let's use a smaller set of diverse points.
        # Maybe 10-20 restarts with 25-50 calls each.
        
        rng = np.random.default_rng(42)
        
        # Define a set of starting points
        starts = []
        starts.append(np.zeros(self.dim)) # Center
        
        # Add random points
        n_random = 15
        for _ in range(n_random):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        # Add some structured points (e.g., mid-points of axes)
        for i in range(self.dim):
            p = np.zeros(self.dim)
            p[i] = 4.0
            starts.append(p)
            p[i] = -4.0
            starts.append(p)

        # Shuffle starts to ensure diversity in early calls? 
        # Actually, for AOCC, we want the best improvement as early as possible.
        # Random points might hit the valley early.
        # Let's just iterate through them.
        
        # Budget per restart
        # Total budget 500.
        # If we have ~20 starts, we can afford ~20-25 calls per start.
        # L-BFGS-B is efficient.
        
        deadline = self.start_time + 280 # Safety margin

        for x0 in starts:
            if self.calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Check if x0 is already better than current best? 
            # We evaluate x0 in the first step of minimize, so it's handled.
            
            # We use a sub-budget for each local run
            # If we have used 100 calls, and have 10 starts left, we have 400/10 = 40 calls each.
            # Dynamic budget allocation:
            remaining_starts = len(starts) - starts.index(x0)
            remaining_calls = self.budget - self.calls
            calls_per_run = max(5, int(remaining_calls / remaining_starts))
            
            # Limit calls_per_run to a reasonable max to avoid one run eating everything
            # L-BFGS-B usually converges in < 50 iterations for 5D smooth functions.
            # Each iteration is 1 function call (gradient approx takes 1 call per dim? No, L-BFGS-B in scipy uses 1 call for f, and approximates gradient. Wait, scipy's L-BFGS-B approximates gradient using finite differences? 
            # Actually, scipy.optimize.minimize with method='L-BFGS-B' does NOT use finite differences for the gradient by default if 'jac' is not provided? 
            # Correction: If 'jac' is not provided, L-BFGS-B in scipy uses a finite difference approximation of the gradient. This requires `dim` additional function calls per iteration!
            # In 5D, that's 1 + 5 = 6 calls per iteration.
            # So 20 iterations = 120 calls.
            # This is expensive.
            
            # If we use 6 calls/iteration, a budget of 30 calls allows only 5 iterations.
            # 5 iterations might not be enough to converge from a random point.
            
            # Alternative: Use a method that doesn't require gradient approx if we can't afford it?
            # Or use fewer, longer runs?
            # If we do 5 restarts, each with 100 calls -> 100/6 = 16 iterations. That's better.
            
            # Let's adjust the number of restarts.
            # Maybe 5-10 restarts is better.
            
            # Let's refine the starts list to be smaller.
            # Keep Center + 9 Random = 10 starts.
            
            # We'll just let the loop handle it. If calls_per_run is too small, it will stop early.
            
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 100, 'maxfun': calls_per_run})
            except Exception:
                pass
                
            if self.calls >= self.budget:
                break

        return self.best_x, self.best_f
