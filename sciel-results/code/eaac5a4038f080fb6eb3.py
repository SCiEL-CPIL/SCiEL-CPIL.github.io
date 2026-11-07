"""Rewritten optimizer for BBOB f23 (Katsuura) using Multi-start Powell's Method."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell's method for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        # Use a deterministic RNG for reproducibility if needed, but random is fine
        rng = np.random.default_rng(42)

        # We will perform multiple local optimizations.
        # Powell's method is very efficient.
        # Each run will use a portion of the budget.
        
        # Strategy: 
        # 1. Initial random points.
        # 2. Run Powell with maxiter such that it uses ~50-100 evals.
        # 3. If we find a very good point, do a final polish with Nelder-Mead.

        # Number of restarts
        # With B=500, d=5.
        # Powell converges in ~d * log(range/precision) iterations?
        # Let's just run until no improvement or budget hit.

        # To ensure we explore, we start with a few random points, 
        # then refine the best ones.

        # Helper to count calls
        def wrapper(x):
            nonlocal calls
            if calls >= B:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Initial population of starting points
        # Use a mix of center and random
        starts = []
        starts.append(np.zeros(d))
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, size=d))
        
        # Sort starts by their initial function value? 
        # This costs 6 calls.
        # Let's just iterate through them.
        
        # We'll do a "greedy" multi-start.
        # Keep a list of "promising" points.
        # Actually, just running Powell from the best current point after each restart is good.
        
        # Let's try a specific schedule:
        # 1. Evaluate 10 random points (10 calls).
        # 2. Take the best one.
        # 3. Run Powell from it, allowing up to 100 calls.
        # 4. If it improved significantly, run another local search from the result.
        # 5. If not, pick the next best from the initial 10 or generate a new random.

        # Simpler robust approach:
        # Loop while budget allows:
        #   Pick a start point (random or best-so-far perturbed)
        #   Run minimize(Powell) with maxiter=20 (approx 100 calls for d=5? No, maxiter is iterations, not function calls. 
        #   For Powell, 1 iteration is d function calls. So maxiter=10 is ~50 calls.)
        #   Update best.
        #   If no improvement, generate new random start.

        # Let's implement this carefully.
        
        current_best_x = None
        current_best_f = np.inf
        
        # Phase 1: Global Exploration
        # Use 50% of budget for exploration?
        # Or just run Powell repeatedly.
        
        # Let's use 10 restarts.
        # Each restart gets ~40 calls.
        
        for i in range(10):
            if calls >= B - 10: # leave some for final polish
                break
            
            # Choose start point
            if i == 0:
                x0 = np.zeros(d)
            else:
                # Random start, or perturbation of best
                if rng.random() < 0.5 and current_best_x is not None:
                    x0 = current_best_x + rng.normal(0, 1.0, size=d)
                else:
                    x0 = rng.uniform(-5, 5, size=d)
            
            x0 = np.clip(x0, -5, 5)
            
            # Run Powell
            # maxiter: number of iterations. For d=5, 1 iter ~ 5-10 calls.
            # Let's allow 20 iterations -> ~100-200 calls? Too many if we do 10 restarts.
            # Let's limit calls per run.
            # We can't easily limit calls in scipy minimize without a wrapper that raises an exception or stops.
            # We can set maxiter low.
            
            # Powell is fast. Let's try maxiter=10.
            res = minimize(wrapper, x0, method='Powell', options={'maxiter': 10, 'xatol': 1e-4})
            
            if res.success or res.fun < current_best_f:
                current_best_f = res.fun
                current_best_x = res.x

        # Phase 2: Local Polish
        if calls < B - 5 and current_best_x is not None:
            # Use Nelder-Mead for final polish
            res = minimize(wrapper, current_best_x, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-10})
            if res.fun < current_best_f:
                current_best_f = res.fun
                current_best_x = res.x

        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
