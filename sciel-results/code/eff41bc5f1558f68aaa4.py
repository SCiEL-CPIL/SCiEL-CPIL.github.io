"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit Lunacek bi-Rastrigin structure: find basin centers, then polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= total_budget or time.time() > deadline:
                return np.inf
            calls_made += 1
            return func(np.clip(x, -5, 5))

        # Phase 1: Coarse grid to locate both basin centers
        # Lunacek has two basins; use a sparse grid in [-4, 4]^5
        # Use a structured grid with step 2.0 to cover the space efficiently
        grid_vals = np.array([-4, -2, 0, 2, 4])
        grid_points = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        for m in range(5):
                            pt = np.array([grid_vals[i], grid_vals[j], grid_vals[k], grid_vals[l], grid_vals[m]])
                            grid_points.append(pt)
        
        # Evaluate all grid points (3125 points is too many, use subset)
        # Use a smarter approach: evaluate along axes and a few key points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(n))
        
        # Axis points at various distances
        for axis in range(n):
            for val in [-4, -3, -2, -1, 1, 2, 3, 4]:
                pt = np.zeros(n)
                pt[axis] = val
                candidates.append(pt)
        
        # Diagonal points
        for val in [-4, -3, -2, -1, 1, 2, 3, 4]:
            candidates.append(np.full(n, val))
        
        # Random points
        for _ in range(50):
            candidates.append(rng.uniform(-4, 4, n))
        
        # Evaluate candidates and keep top 20
        cand_results = []
        for pt in candidates:
            if calls_made >= total_budget or time.time() > deadline:
                break
            fval = safe_func(pt)
            if fval < np.inf:
                cand_results.append((fval, pt.copy()))
        
        cand_results.sort(key=lambda x: x[0])
        
        # Phase 2: For each of the top candidates, do local refinement
        top_k = min(10, len(cand_results))
        for i in range(top_k):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            f0, x0 = cand_results[i]
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Coordinate descent polish
            x = x0.copy()
            fx = f0
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                improved = True
                while improved:
                    improved = False
                    for dim_idx in range(n):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        # Try positive and negative steps
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[dim_idx] += direction * step
                            ft = safe_func(x_trial)
                            if ft < fx:
                                x = x_trial
                                fx = ft
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                        # Also try zero (stay)
        
        # Phase 3: Final fine-tuning with smaller steps around best
        if calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001]
            for step in fine_steps:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                for _ in range(3):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    for dim_idx in range(n):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[dim_idx] += direction * step
                            ft = safe_func(x_trial)
                            if ft < fx:
                                x = x_trial
                                fx = ft
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
