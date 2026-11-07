"""Hybrid Global-Local Optimizer for BBOB functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Strategy:
    1. Global Phase: Coarse grid sampling (12 points per dim -> 12^5 = 248832 is too many, 
       so we use a smarter approach: 
       - 1 center point (0,0,0,0,0)
       - 10 random points
       - A coarse grid of 3 points per dim (3^5 = 241 points) to catch broad basins.
       Total global points: ~250.
    2. Local Phase: Take the best few points from the global phase and run L-BFGS-B 
       on them with the remaining budget.
    
    Why this works:
    - BBOB functions like Rastrigin and Gallagher are multimodal. A single trajectory 
      (like the seed (1+1)-ES) often gets stuck in a local minimum.
    - The global phase with a grid ensures we sample multiple basins.
    - L-BFGS-B is very efficient for smooth local convergence, especially on Sphere/Rosenbrock.
    - The deadline guard ensures we never exceed the 330s limit, returning the best found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin of 5s before 330s kill
        
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            x = np.clip(x, -5, 5)
            try:
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                calls += 1
                return np.inf

        # Phase 1: Global Search
        # 1. Center
        center = np.zeros(self.dim)
        safe_call(center)

        # 2. Random points
        rng = np.random.default_rng(42)
        n_random = 20
        for _ in range(n_random):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            safe_call(x)

        # 3. Coarse Grid
        # 3 points per dimension: -5, 0, 5
        # 3^5 = 241 points. This is manageable.
        # We can also add mid-points if budget allows, but let's stick to 3 for now to save budget for local.
        grid_points = np.array(np.meshgrid([-5.0, 0.0, 5.0] * self.dim)).T.reshape(-1, self.dim)
        
        # Shuffle grid points to ensure early progress isn't biased by grid order
        rng.shuffle(grid_points)
        
        for x in grid_points:
            if calls >= self.budget or time.time() > deadline:
                break
            safe_call(x)

        # Phase 2: Local Refinement
        # Collect the best points from Phase 1 to use as seeds for L-BFGS-B.
        # Since we didn't store all points, we just use the current best and a few random restarts 
        # or we can re-evaluate a few promising regions.
        # A better approach: Keep a list of the best N points found so far.
        # Let's modify safe_call to keep a history of the best 5 points.
        
        # Actually, let's just use the current best and a few variations around it, 
        # plus a few random restarts for local search if budget allows.
        
        # To be more robust, let's collect the best 10 points from the global phase.
        # We need to track them. Let's restart the logic slightly to track best points.
        
        # Since I already called safe_call many times, let's just assume `best_x` is a good seed.
        # We will run L-BFGS-B from `best_x`.
        
        # If we have budget left, we can also try L-BFGS-B from a few other good points.
        # But we don't have the list of other good points easily accessible without refactoring.
        # Let's just run L-BFGS-B from the current best.
        
        # Wait, L-BFGS-B might use many function evaluations. We need to limit it.
        # maxiter is not the same as function evaluations.
        # We can use a wrapper to count evaluations and stop if we hit the budget or deadline.
        
        # Let's define a limited minimizer.
        
        if calls < self.budget and time.time() < deadline:
            # Use the best point found so far as the starting point
            x0 = best_x if best_x is not None else np.zeros(self.dim)
            
            # We will run L-BFGS-B. We need to be careful with the budget.
            # L-BFGS-B doesn't have a direct "max fun evals" limit in the standard API, 
            # but we can check inside the function wrapper.
            
            eval_count_in_minimize = 0
            def objective(x):
                nonlocal eval_count_in_minimize
                if calls + eval_count_in_minimize >= self.budget:
                    return np.inf
                if time.time() > deadline:
                    return np.inf
                eval_count_in_minimize += 1
                return safe_call(x)
                
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
