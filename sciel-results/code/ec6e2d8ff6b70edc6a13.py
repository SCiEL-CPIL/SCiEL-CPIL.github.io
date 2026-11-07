import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid search to identify promising basins
        # Lunacek bi-Rastrigin has two basins. A coarse grid helps find both.
        # We'll use a 3x3x3x3x3 grid = 243 points. This is feasible within budget.
        # But 243 calls is a lot. Let's try a sparser approach first or use the grid.
        # Actually, let's do a smarter initialization.
        
        # Let's use a population-based CMA-ES or similar, but simpler.
        # Let's try a hybrid: Grid search for initialization, then local search.
        
        # Grid points
        # 5 dimensions. 3 points per dim = 243 points.
        # If budget is 500, we have ~250 left for local search.
        
        points = np.array(np.meshgrid(*[np.linspace(-5, 5, 3) for _ in range(dim)])).T.reshape(-1, dim)
        
        # Shuffle to ensure we don't bias towards one corner
        idx = rng.permutation(len(points))
        points = points[idx]
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Evaluate grid
        for p in points:
            if calls >= budget - 10: # Keep 10 for safety
                break
            if time.time() > deadline:
                break
            f_val = func(p)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # Now perform local search from the best point found
        # Use a simple Nelder-Mead or a random walk with decreasing step size
        # Let's use a random walk with adaptive step size
        current_x = best_x.copy()
        current_f = best_f
        step_size = 1.0
        
        # Remaining budget
        remaining = budget - calls
        
        # Local search loop
        while calls < budget and time.time() < deadline:
            # Generate a random perturbation
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            
            new_x = current_x + step_size * direction
            new_x = np.clip(new_x, -5.0, 5.0)
            
            f_new = func(new_x)
            calls += 1
            
            if f_new < current_f:
                current_x = new_x
                current_f = f_new
                # Increase step size slightly on success
                step_size *= 1.1
            else:
                # Decrease step size on failure
                step_size *= 0.9
            
            # Clamp step size
            step_size = np.clip(step_size, 1e-5, 5.0)
            
            # If step size is very small, we've converged locally
            if step_size < 1e-4:
                break

        return current_x, current_f
