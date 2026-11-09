"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical global-to-local search for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Phase 1: Coarse grid scan to find the global basin
        # Use ~400 calls for a structured grid over [-5, 5]^5
        # We'll use a Latin hypercube-like sampling for better coverage
        grid_calls = 400
        
        # Generate grid points: 4 points per dimension gives 4^5 = 1024, too many
        # Use 3 points per dimension: 3^5 = 243 points, plus some random
        # Actually, let's use a smarter approach: stratified sampling
        n_per_dim = 3
        grid_points = []
        for i in range(n_per_dim ** dim):
            point = []
            temp = i
            for d in range(dim):
                point.append(-5 + (temp % n_per_dim) * (10.0 / (n_per_dim - 1)))
                temp //= n_per_dim
            grid_points.append(np.array(point))
        
        # Shuffle grid points to avoid systematic bias
        rng.shuffle(grid_points)
        
        # Evaluate grid points
        for x in grid_points:
            if calls_made >= grid_calls or time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
        
        # Phase 2: Localized CMA-ES-like refinement around the best grid point
        # Use remaining ~100 calls
        if calls_made < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with small step size since we're in a basin
            stall_count = 0
            max_stall = 20
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                # Generate candidate with adaptive step size
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Accept improvement
                    x, fx = y, fy
                    sigma *= 1.05
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    # Reject, adapt step size
                    sigma *= 0.95
                    stall_count += 1
                    
                # If stalled too much, try a larger perturbation
                if stall_count > max_stall:
                    sigma = 1.0
                    stall_count = 0
                    # Occasionally try a random restart from the best point
                    if rng.random() < 0.3:
                        x = np.clip(best_x + rng.standard_normal(dim) * 0.3, -5, 5)
                        fx = func(x)
                        calls_made += 1
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
