"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell local search with robust budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        # Call counter
        call_count = 0
        max_calls = self.budget
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Multi-start strategy
        # We will try several random starting points.
        # To improve early progress, we start with a few points near the center and corners.
        # Then random points.
        
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Corners (scaled to be inside)
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        # If we have budget, add more random starts
        # Powell is efficient, so we can afford ~5-10 runs
        num_starts = 10
        
        for i in range(num_starts):
            if call_count >= max_calls:
                break
            if time.time() > deadline:
                break
                
            if i < len(starts):
                x0 = starts[i]
            else:
                x0 = rng.uniform(-5, 5, self.dim)
                
            # Run Powell
            # maxiter is a heuristic; we rely on call count
            try:
                res = minimize(counted_func, x0, method='Powell', 
                               options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If an exception occurs (e.g. budget exceeded), we stop
                break

        # If no successful optimization, return the first point or best found
        if best_x is None or np.all(best_x == 0):
             # If we never updated best_x, we need to ensure we have at least one valid evaluation
             # However, the loop above likely evaluated at least the center point.
             # If best_f is still inf, it means no calls were made or all failed?
             # But counted_func raises on budget, and minimize catches it?
             # Actually, if minimize raises, we break.
             # If we didn't break, res.fun should be set.
             # Let's just ensure best_x is valid.
             if best_f == float('inf'):
                 try:
                     best_f = counted_func(best_x)
                 except:
                     pass

        return best_x, best_f
# EVOLVE-BLOCK-END
