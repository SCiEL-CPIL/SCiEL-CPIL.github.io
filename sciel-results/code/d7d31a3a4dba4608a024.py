"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        # Coarse grid of initial points
        # 3 points per dimension -> 3^5 = 243 starts
        # This is well within the budget of 500 calls if each L-BFGS-B run
        # takes ~1 call (which is unlikely, it takes more, but we cap total calls).
        
        # Actually, L-BFGS-B with finite differences uses (dim+1) calls per iteration.
        # In 5D, that's 6 calls/iteration.
        # Budget 500 / 6 = ~83 iterations total across all runs.
        # If we have 243 starts, we can't afford many iterations per start.
        # We need to be smarter about the number of starts or use a cheaper local method.
        
        # Alternative: Use Nelder-Mead? It uses dim+1 = 6 calls/iteration too.
        # But it doesn't need gradients (which L-BFGS-B approximates).
        
        # Let's stick to L-BFGS-B but limit the number of starts to ensure
        # we have enough budget for refinement.
        # Or, better: use a smaller grid. 2 points per dim -> 32 starts.
        # Or 3 points per dim but stop early if no improvement.
        
        # Let's try a hybrid:
        # 1. Evaluate a coarse grid (e.g. 3^5 = 243 points). Cost: 243 calls.
        # 2. Pick the best K points (e.g. K=5).
        # 3. Run L-BFGS-B from those K points with the remaining budget.
        
        # Total calls: 243 + 5 * (remaining/6).
        # Remaining budget: 500 - 243 = 257.
        # 257 / 6 = 42 iterations per start.
        # This seems reasonable.
        
        points = np.array(np.meshgrid(*[np.linspace(-4, 4, 3) for _ in range(self.dim)])).T.reshape(-1, self.dim)
        
        best_x = None
        best_f = float('inf')
        call_count = 0
        
        # Phase 1: Coarse Grid Evaluation
        for p in points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                fp = func(p)
                call_count += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p
            except Exception:
                pass
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Select top few points from the grid to start local optimization
        # We didn't store all points, so let's just use the best one found,
        # or re-evaluate a few neighbors?
        # Better: keep track of top K during grid search.
        
        # Let's refine the logic to keep top K.
        top_k = []
        K = 5
        
        # Re-do Phase 1 with top K tracking
        best_x = None
        best_f = float('inf')
        call_count = 0
        top_k = []
        
        for p in points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                fp = func(p)
                call_count += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p
                top_k.append((fp, p))
            except Exception:
                pass
        
        # Sort top_k and take the best K
        top_k.sort(key=lambda x: x[0])
        start_points = [p for f, p in top_k[:K]]
        
        # Phase 2: L-BFGS-B
        # Wrap func to count calls and handle deadline
        def wrapped_func(x):
            nonlocal call_count
            if call_count >= self.budget or time.time() > deadline:
                return float('inf')
            try:
                val = func(x)
                call_count += 1
                return val
            except Exception:
                return float('inf')
        
        for sp in start_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            try:
                res = minimize(wrapped_func, sp, method='L-BFGS-B', bounds=[(-5, 5)] * self.dim, options={'maxiter': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
