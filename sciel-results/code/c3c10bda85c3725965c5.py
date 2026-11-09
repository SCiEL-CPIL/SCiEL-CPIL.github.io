"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer:
    1. Coarse grid search to quickly locate the basin of attraction.
    2. (1+1)-ES with 1/5 success rule for global/local refinement.
    3. Nelder-Mead polish for high-precision final convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 295s to be safe
        deadline = start_time + 295.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def remaining():
            return budget - calls_made

        def do_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # --- Phase 1: Coarse Grid Search ---
        # f6 (Attractive Sector) often has a broad basin.
        # A coarse grid helps avoid getting stuck in bad local minima
        # if the random start of ES is far from the optimum.
        # We use a small grid to keep call count low (e.g., 3^5 = 243 points).
        # However, 243 calls is ~50% of budget. Let's be sparser.
        # Let's try a grid of size 2 or 3 in each dimension.
        # Size 2: 2^5 = 32 points. Very cheap.
        # Size 3: 3^5 = 243 points. Might be too expensive if we want to save for ES.
        # Let's use a hybrid: evaluate corners and center, plus a few random points?
        # Or just a very sparse grid: {-4, 0, 4} for each dim? 3^5 = 243.
        # If budget is 500, 243 calls leaves 257 for ES. This is reasonable.
        # Let's try {-3, 0, 3} to be slightly more central.
        
        grid_vals = [-3.0, 0.0, 3.0]
        # Generate grid points
        import itertools
        grid_points = list(itertools.product(grid_vals, repeat=dim))
        
        # Shuffle to avoid deterministic bias if function has symmetries?
        # rng.shuffle(grid_points) # grid_points is a list of tuples, need to convert to list of arrays or shuffle indices
        indices = np.arange(len(grid_points))
        rng.shuffle(indices)
        
        # Evaluate grid points
        for idx in indices:
            if not time_ok() or calls_made >= budget:
                break
            pt = np.array(grid_points[idx], dtype=float)
            do_call(pt)
            
        # If we didn't make any calls (shouldn't happen), start at 0
        if best_x is None:
            do_call(np.zeros(dim))

        # --- Phase 2: (1+1)-ES ---
        # Start ES from the best point found in the grid
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma: large enough to explore, but we already did coarse search.
        # Maybe start smaller? 2.0 is a good default.
        sigma = 2.0
        
        # 1/5 success rule parameters
        successes = 0
        total = 0
        window_size = 10

        while calls_made < budget and time_ok():
            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = do_call(y)
            
            if fy == np.inf: # Budget or time exhausted
                break

            if fy < fx:
                x, fx = y, fy
                successes += 1
            # else: keep x

            total += 1

            # Update sigma every window_size steps using 1/5 success rule
            if total % window_size == 0:
                success_rate = successes / total
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.87
                successes = 0
                total = 0

            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        # --- Phase 3: Nelder-Mead Polish ---
        # Use remaining budget for a final high-precision local search.
        # Nelder-Mead is effective for smooth functions like BBOB f6.
        remaining = remaining()
        if remaining > 10 and time_ok():
            try:
                from scipy.optimize import minimize
                
                polish_calls = [0]
                max_polish_calls = min(remaining, 200)
                
                def polish_func(z):
                    if polish_calls[0] >= max_polish_calls:
                        return best_f
                    polish_calls[0] += 1
                    z_clipped = np.clip(z, -5, 5)
                    return func(z_clipped)

                # Nelder-Mead with small initial simplex
                initial_step = 0.1
                # Construct initial simplex
                simplex = [best_x + initial_step * e for e in np.eye(dim)]
                simplex = np.array(simplex)
                
                result = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_polish_calls,
                        'initial_simplex': simplex,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun
            except Exception:
                pass # If polish fails, keep the ES result

        return best_x, best_f
# EVOLVE-BLOCK-END
