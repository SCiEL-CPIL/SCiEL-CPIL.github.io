"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Hybrid Multi-start L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_used = 0
        
        # Wrapped function to track calls and enforce budget
        def wrapped_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= self.budget:
                return 1e10 # Return high value to stop optimizer if budget hit
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # 1. Initial Point Generation
        # We use a mix of the origin (common for Rosenbrock) and random points
        # to ensure we don't get stuck in a local minimum if the shift is far.
        
        # Deterministic starting points
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 5.0)
        starts.append(np.ones(self.dim) * -5.0)
        
        # Random starting points
        rng = np.random.default_rng(42)
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle starts to interleave deterministic and random
        # Actually, for Rosenbrock, origin is often a good start if shift is small.
        # But we don't know the shift. Let's just iterate through them.
        
        # Sort by some heuristic? No, just iterate. 
        # To maximize early progress (AOCC), we should try the most likely candidates first.
        # Origin is a classic trap but also the global min for standard Rosenbrock.
        # Given the "shifted" nature, random is safer, but origin is cheap.
        
        # Let's prioritize:
        # 1. Origin (0,0,0,0,0)
        # 2. A few random points
        # 3. More random points
        
        # Re-order: Origin first, then random
        starts = [np.zeros(self.dim)] + [s for s in starts if not np.allclose(s, 0)]
        
        for x0 in starts:
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Update best if the local minimum is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # If we have remaining budget, we could do a final polish or more random restarts
        # but L-BFGS-B is usually very efficient. 
        # If best_x is None (shouldn't happen), fall back to random
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f

# EVOLVE-BLOCK-END
