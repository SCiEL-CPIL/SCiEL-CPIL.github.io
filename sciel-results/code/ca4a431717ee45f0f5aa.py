"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # We have 500 calls. Nelder-Mead in 5D takes roughly 5-10 * dim * iterations.
        # Let's allocate calls for multiple restarts.
        # Strategy: 
        # 1. Initial coarse sampling to find promising basins.
        # 2. Local refinement using Nelder-Mead.
        # 3. If time/budget allows, restart from best found or new random points.
        
        # Since we don't know the landscape, a mix of random and grid-based starts is good.
        # BBOB f22 has 21 peaks. The global minimum is one of them.
        # Random starts with a decent probability hit the basin of attraction.
        
        def count_calls(x):
            nonlocal self
            self.calls_made += 1
            return func(x)

        # We will use a simple loop of Nelder-Mead runs.
        # Each run consumes budget.
        # Nelder-Mead options: maxiter, maxfev.
        
        # Let's try 5-10 restarts.
        # Each restart:
        # 1. Pick a starting point (random or based on previous best).
        # 2. Run Nelder-Mead with maxfev limited to ensure we finish.
        
        # To be safe with the "500 calls" hard limit, we must strictly count.
        # SciPy's minimize does not natively support a global call counter easily 
        # unless we wrap the function, which we are doing.
        
        # However, if we exceed 500 calls, the result is invalid.
        # So we must stop calling func if self.calls_made >= 500.
        
        # Let's structure the search:
        # Phase 1: 20 random starts with small Nelder-Mead budgets (e.g., 10-20 calls each).
        # Phase 2: Take the best 2-3 points and run longer Nelder-Mead.
        
        # Actually, Nelder-Mead can be inefficient if not tuned.
        # Powell's method is often more robust for smooth functions without gradient.
        # Let's use Powell's method. It is derivative-free and handles local minima well.
        
        # Powell's method in 5D:
        # Each iteration moves along 5 directions + 1 new direction = 6 function calls per iteration?
        # Roughly O(dim) calls per iteration.
        
        # Let's allocate 10 restarts.
        # Budget per restart: ~40 calls.
        
        rng = np.random.default_rng(42)
        
        # Generate a set of starting points
        # Mix of random and grid-like
        starts = []
        # 5 random points
        for _ in range(5):
            x0 = rng.uniform(-5, 5, size=self.dim)
            starts.append(x0)
            
        # 5 points from a coarse grid (3^5 = 243, pick 5 spread out)
        # Let's just use random for simplicity and robustness to hidden shifts.
        # But maybe centering around 0 is good? BBOB instances are shifted.
        # Random in [-5, 5] is safe.
        
        # Let's do 8 restarts.
        for i in range(8):
            if self.calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Choose start point
            if i < len(starts):
                x0 = starts[i]
            else:
                # If we need more restarts, pick random
                x0 = rng.uniform(-5, 5, size=self.dim)
                
            # Remaining budget for this run
            remaining = self.budget - self.calls_made
            # Limit maxfev to not overshoot
            # Powell's method might need a few iterations.
            # Let's give it up to 20% of remaining budget, but at least 10 calls.
            maxfev = max(10, int(remaining * 0.2))
            
            # Run Powell
            try:
                res = minimize(
                    count_calls, 
                    x0, 
                    method='Powell', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'maxfev': maxfev, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except Exception:
                pass
                
            # Optional: If we found a very good point, maybe do a quick local polish?
            # Powell already does local polish.
            
        # If we have calls left, do a final random restart from the best point with perturbation?
        # Or just stop.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
