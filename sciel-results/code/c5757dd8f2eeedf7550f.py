"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Global Lattice Search with Best-First Expansion (FPS) followed by Local Refinement.
    
    Strategy:
    1. Global Exploration (FPS): 
       - Generate a pool of random candidates.
       - Iteratively select the candidate that is farthest from the set of already evaluated points.
       - This ensures uniform coverage of the [-5, 5]^5 space, helping to locate the correct 
         basin of attraction for multimodal functions (Rastrigin, Gallagher).
    2. Local Exploitation (1+1-ES):
       - Once a sufficient number of global points have been sampled (e.g., 50-100),
         switch to a local (1+1)-Evolution Strategy starting from the best point found.
       - This allows for rapid convergence and fine-tuning, which is crucial for AOCC scoring 
         that rewards early and sustained progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Phase 1: Global Exploration using Farthest Point Sampling (FPS)
        # We use a pool of candidates and pick the one farthest from the current set.
        # This is O(N * PoolSize * Dim) per step if done naively, but we can optimize 
        # by pre-computing or just keeping the pool small enough.
        # Let's use a pool of 1000 points.
        
        # Initial seed points to anchor the search
        initial_points = [
            np.zeros(n),
            np.ones(n) * 5.0,
            np.ones(n) * -5.0,
            np.ones(n) * 3.0,
            np.ones(n) * -3.0,
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.ones(n) * 1.0,
            np.ones(n) * -1.0,
            rng.uniform(-5, 5, n)
        ]
        
        # Generate a pool of random candidates
        pool_size = 2000
        pool = rng.uniform(-5, 5, (pool_size, n))
        
        evaluated_points = []
        
        # Evaluate initial points
        for x in initial_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            evaluated_points.append(x.copy())
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # FPS Loop
        # We want to use about 40% of the budget for global exploration
        n_global = max(20, int(0.4 * total_budget))
        
        # Convert evaluated_points to a numpy array for distance calculation
        # We'll maintain a list and compute min dist to all previous points.
        # To speed this up, we can maintain a "min_dist_to_set" array for the pool.
        # min_dist[i] = min distance from pool[i] to any point in evaluated_points
        
        # Initialize min_dist for the pool
        # Distance from each pool point to the first evaluated point
        if evaluated_points:
            first_point = evaluated_points[0]
            min_dists = np.linalg.norm(pool - first_point, axis=1)
        else:
            min_dists = np.full(pool_size, np.inf)
            
        # Add first point to evaluated
        # (Already done above, but we need to update min_dists for subsequent points)
        
        for _ in range(n_global - len(initial_points)):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Find the point in the pool with the maximum min_dist
            idx = np.argmax(min_dists)
            x = pool[idx]
            
            # Evaluate it
            fx = func(x)
            calls_made += 1
            evaluated_points.append(x.copy())
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Update min_dists for the remaining pool points
            # We need to remove the selected point from the pool effectively.
            # We can set its min_dist to -1 so it's not picked again.
            min_dists[idx] = -1
            
            # Update min_dists for other points: min(min_dists[j], dist(pool[j], x))
            # dist(pool[j], x) for all j
            dists_to_new = np.linalg.norm(pool - x, axis=1)
            min_dists = np.minimum(min_dists, dists_to_new)
            # Ensure we don't pick -1 points again (though min with -1 stays -1)
            # Actually, if min_dists was -1, it stays -1. If it was positive, it might become smaller.
            # We want to ignore points already picked.
            # The argmax will naturally avoid -1 if there are any positive values left.
            # If all are -1, we break or handle edge case.
            if np.max(min_dists) < 0:
                break

        # Phase 2: Local Refinement using (1+1)-ES
        if best_x is None:
            # Fallback
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()
        fx = best_f
        
        # Initial step size for local search
        # Since we've done global search, we can start with a smaller sigma
        sigma = 1.0
        
        sigma_inc = 1.2
        sigma_dec = 0.8
        sigma_min = 1e-5
        sigma_max = 5.0

        no_improve_counter = 0
        stall_threshold = 20
        restart_count = 0
        max_restarts = 10

        while calls_made < total_budget:
            if time.time() > deadline:
                break

            # Generate offspring
            noise = rng.standard_normal(n)
            y = x + sigma * noise
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= sigma_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    no_improve_counter = 0
            else:
                sigma *= sigma_dec
                no_improve_counter += 1

            sigma = max(sigma_min, min(sigma, sigma_max))

            # Restart if stalled
            if no_improve_counter > stall_threshold and restart_count < max_restarts:
                restart_count += 1
                no_improve_counter = 0
                
                # Restart from a point near the best, or a random point if very stalled
                if restart_count % 2 == 0:
                    x = best_x + 0.5 * rng.standard_normal(n)
                else:
                    x = rng.uniform(-5, 5, n)
                
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
