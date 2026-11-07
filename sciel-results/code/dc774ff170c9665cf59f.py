"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Multi-Start Nelder-Mead Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10])
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f

        # 1. Initial Sampling: Spread points across the domain to cover the front
        n_init = 30
        # Use a mix of random and grid-like points for initial coverage
        # We use np.random which is seeded by the evaluator
        init_X = []
        # 15 random points
        for _ in range(15):
            init_X.append(lo + span * np.random.random(dim))
        # 15 points on a grid (2^4 = 16 corners, take 15)
        # Actually, for dim=4, a full factorial of 2 levels is 16 points.
        # Let's use 3 levels per dim? No, too many.
        # Let's use random + some structured mid-points.
        # 4 mid-points (varying one dim at a time)
        for i in range(dim):
            pt = (lo + hi) / 2.0
            pt[i] = lo[i] + 0.25 * span[i]
            init_X.append(pt)
        for i in range(dim):
            pt = (lo + hi) / 2.0
            pt[i] = lo[i] + 0.75 * span[i]
            init_X.append(pt)
        
        # Evaluate initial points
        for x in init_X:
            f = eval_point(x)
            if f is None:
                break

        # 2. Multi-Start Local Search
        # We will pick starting points from the current non-dominated set (or random if empty)
        # and run a short Nelder-Mead optimization on a weighted scalarization.
        # To ensure coverage, we cycle through different weights.
        
        # Weights for scalarization: w1 * f1 + w2 * f2
        # We want to explore the whole front.
        # We will perform several "rounds".
        
        # Estimate remaining budget for local searches
        # Each Nelder-Mead run with dim=4 takes roughly 2*dim + iterations evaluations.
        # Let's aim for ~5-10 evaluations per local search to keep it "multi-start" and not "single deep".
        # Actually, Nelder-Mead on 4D usually converges in 20-50 evals.
        # If we use 200 evals total, and 30 are init, we have 170.
        # 170 / 10 = 17 local searches.
        
        max_evals_per_search = 15
        n_searches = (budget - used) // max_evals_per_search
        
        # Weights: We want to cover the front. 
        # Uniform distribution of weights in [0, 1]
        # Or specific weights to target different parts.
        # Let's use a set of predefined weights + random ones.
        
        weights_list = []
        # Add some deterministic weights to ensure coverage
        weights_list.append(np.array([0.5, 0.5]))
        weights_list.append(np.array([0.1, 0.9]))
        weights_list.append(np.array([0.9, 0.1]))
        weights_list.append(np.array([0.25, 0.75]))
        weights_list.append(np.array([0.75, 0.25]))
        
        # Fill rest with random weights
        while len(weights_list) < n_searches:
            w1 = np.random.random()
            weights_list.append(np.array([w1, 1.0 - w1]))

        # Shuffle weights to mix exploration
        np.random.shuffle(weights_list)
        
        # Starting points:
        # Ideally, start local searches from good points.
        # We will maintain a list of "candidates" for starting points.
        # Initially, all evaluated points are candidates.
        # As we find better points, we add them.
        
        # Simple selection: Pick a random point from the current best non-dominated set.
        # If the set is large, pick randomly to maintain diversity.
        
        current_F = np.array(all_F) if all_F else np.empty((0, 2))
        current_X = np.array(all_X) if all_X else np.empty((0, 4))

        def get_non_dominated_indices(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            n = len(F)
            # Vectorized non-dominated check
            # F[:, None, :] <= F[None, :, :]
            le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        for i in range(n_searches):
            if used >= budget or time.time() > deadline:
                break
            
            w = weights_list[i % len(weights_list)]
            
            # Select starting point
            nd_idx = get_non_dominated_indices(current_F)
            if len(nd_idx) > 0:
                # Pick a random non-dominated point
                start_idx = np.random.choice(nd_idx)
                x0 = current_X[start_idx].copy()
            else:
                # Pick a random point from all
                start_idx = np.random.choice(len(current_F))
                x0 = current_X[start_idx].copy()
            
            # Define scalarized objective
            def obj(x):
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    return 1e10
                return float(np.dot(w, f))
            
            # We need to track evaluations inside minimize to count towards budget
            # SciPy's minimize does not have a direct callback for eval count in all versions/methods easily accessible
            # for stopping based on a global counter without wrapping.
            # We will wrap the objective.
            
            local_eval_count = [0]
            def wrapped_obj(x):
                nonlocal used
                if used >= budget:
                    return 1e10
                if time.time() > deadline:
                    return 1e10
                local_eval_count[0] += 1
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    return 1e10
                # Record this evaluation
                all_X.append(x)
                all_F.append(f)
                used += 1
                return float(np.dot(w, f))

            # Run Nelder-Mead
            # Options: maxiter limits iterations, but we want to limit function calls.
            # Nelder-Mead in scipy doesn't have a direct maxfev option in all versions, 
            # but we can stop early if we hit our budget in the wrapper.
            # To be safe, we'll catch the return and check used.
            
            try:
                res = minimize(wrapped_obj, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 50})
            except Exception:
                pass
            
            # Update current_F and current_X with new points
            # We need to append the new points to the arrays used for selection
            # all_X and all_F are lists, so we need to convert or keep track.
            # For efficiency, we'll just re-create the arrays if we are running many searches,
            # but with ~17 searches and ~200 points, it's fast enough.
            
            # The new points are already in all_X and all_F
            # We need to update current_F and current_X to include them for the next iteration's selection
            # Actually, we can just use all_F and all_X directly in get_non_dominated_indices
            # but that gets expensive. 
            # Let's just append the new batch to current_F/X
            
            # Determine how many new points were added
            # We don't have a precise count easily without diffing, but we can just 
            # assume the last `local_eval_count[0]` entries are new.
            # Wait, `wrapped_obj` appends to all_X/all_F.
            # So we can just slice.
            
            if local_eval_count[0] > 0:
                new_X = np.array(all_X[-local_eval_count[0]:])
                new_F = np.array(all_F[-local_eval_count[0]:])
                current_X = np.vstack([current_X, new_X])
                current_F = np.vstack([current_F, new_F])

        return all_X
# EVOLVE-BLOCK-END
