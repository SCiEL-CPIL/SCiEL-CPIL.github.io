"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global Lattice Search with Best-First Expansion (Farthest Point Sampling)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s/330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls_used < budget and time.time() <= deadline

        # Initialize with a center point
        x0 = np.zeros(dim)
        f0 = safe_call(x0)
        if np.isfinite(f0):
            update_best(x0, f0)
        
        # List of evaluated points: (x, f, min_dist_to_other_evaluated_points)
        # We will maintain a list of points and their minimum distance to any other evaluated point.
        # To select the next point, we find the one with the maximum min_dist.
        # This is Farthest Point Sampling (FPS).
        
        # For efficiency with 300 calls, O(N^2) distance calculation is fine (300^2 = 90k ops).
        
        evaluated_points = [x0.copy()]
        if np.isfinite(f0):
            # Store (x, f, min_dist). Initially only one point, min_dist is inf or 0?
            # FPS logic: 
            # 1. Start with one point.
            # 2. For all other points in the candidate pool (or generate new ones?), 
            #    calculate distance to the nearest evaluated point.
            #    Select the one with max distance.
            
            # We need a source of candidate points. 
            # Strategy: 
            # 1. Generate a coarse grid or random pool?
            # 2. Or generate candidates on the fly?
            #    "Global Lattice Search" suggests a structured grid.
            #    Let's create a hierarchical grid or a large random pool and select from it.
            #    Given the budget is small (300), a random pool of ~1000 points is cheap to generate.
            #    Then we select from this pool using FPS.
            
            # Let's generate a pool of candidates.
            # A mix of random points and grid points might be good.
            # Let's stick to a large random pool for simplicity and coverage.
            n_pool = 2000
            pool = rng.uniform(-5, 5, (n_pool, dim))
            
            # To ensure we don't pick the same point twice, we track evaluated indices or just check distance.
            # Since it's continuous, exact duplicates are unlikely with random generation, 
            # but we should avoid re-evaluating a point very close to an existing one if possible,
            # although FPS naturally handles this by picking far points.
            
            # We will maintain a list of (index, min_dist) for the pool.
            # min_dist[i] = min(distance(pool[i], evaluated_points[j]) for all j)
            
            # Initialize min_dist for all pool points
            min_dists = np.full(n_pool, np.inf)
            
            # Update min_dists with the first evaluated point
            # dists = np.linalg.norm(pool - x0, axis=1)
            # min_dists = np.minimum(min_dists, dists)
            # Actually, we can just compute this incrementally.
            
            # Current evaluated points list
            eval_xs = [x0]
            
            # Calculate initial distances
            min_dists = np.linalg.norm(pool - x0, axis=1)
            
            # Main loop
            while time_ok():
                # Find the point in the pool with the maximum min_dist
                # We should exclude points that are already evaluated? 
                # With random generation, it's unlikely we pick the exact same float.
                # But to be safe, we can mark selected points.
                # However, if we pick a point, we add it to eval_xs.
                # The next min_dist calculation will be small for points near it.
                
                # Optimization: We can just find the argmax of min_dists.
                # But we must ensure we don't pick a point that is effectively the same as an existing one.
                # The FPS criterion handles this: if a point is close to an existing one, its min_dist is small.
                
                idx_max = np.argmax(min_dists)
                x_new = pool[idx_max]
                f_new = safe_call(x_new)
                
                if np.isfinite(f_new):
                    update_best(x_new, f_new)
                    eval_xs.append(x_new)
                    
                    # Update min_dists for all remaining pool points
                    # dist_new = np.linalg.norm(pool - x_new, axis=1)
                    # min_dists = np.minimum(min_dists, dist_new)
                    
                    # Efficient update:
                    dist_new = np.linalg.norm(pool - x_new, axis=1)
                    min_dists = np.minimum(min_dists, dist_new)
                    
                    # Optional: Mark the selected point as "used" to avoid re-selecting it if it somehow has high dist?
                    # No, its dist to itself is 0, so min_dists[idx_max] becomes 0. It won't be selected again.
                else:
                    # If call failed or returned inf, we still add it to evaluated to avoid retrying?
                    # Or just skip? Skipping might cause us to pick the same "far" point again if it fails again?
                    # Better to mark it as evaluated (dist 0) so we move on.
                    eval_xs.append(x_new)
                    dist_new = np.linalg.norm(pool - x_new, axis=1)
                    min_dists = np.minimum(min_dists, dist_new)

            # After FPS, we have a set of well-spread points.
            # We can now refine the best point found using a local method.
            if best_x is not None and time_ok():
                # Local refinement on the global best
                # Use a simple random hill climbing or (1+1) ES with small steps
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                # Use remaining budget for refinement
                # Estimate remaining calls
                remaining = budget - calls_used
                for _ in range(remaining):
                    if not time_ok():
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = safe_call(y)
                    if fy < fx:
                        x, fx = y, fy
                        update_best(x, fx)
                        sigma *= 1.1 # Expand slightly on success
                    else:
                        sigma *= 0.9 # Shrink on failure
                    if sigma < 1e-5:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0
        return best_x, best_f
# EVOLVE-BLOCK-END
