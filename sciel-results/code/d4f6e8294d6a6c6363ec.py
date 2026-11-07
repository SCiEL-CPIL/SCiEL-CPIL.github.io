import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Evaluation tracking
        eval_count = 0
        X_samples = []
        F_samples = []

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # Phase 1: Stratified Random Sampling (Latin Hypercube)
        # Use 15% of budget for initial diverse sampling
        n_random = min(int(budget * 0.15), 30)
        
        # Simple LHS: divide range into n_random bins, pick one random point in each bin for each dimension
        X_lhs = np.zeros((n_random, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], n_random + 1)
            for i in range(n_random):
                X_lhs[i, d] = edges[i] + rng.random() * (edges[i+1] - edges[i])
        
        for i in range(n_random):
            if not time_ok():
                break
            x = X_lhs[i]
            f = evaluate(x)
            X_samples.append(x)
            F_samples.append(f)

        # Helper to find non-dominated points
        def get_nondominated_indices(F):
            if len(F) == 0:
                return []
            F = np.array(F)
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        # Phase 2: Multi-start Nelder-Mead Local Search
        # Generate diverse weight vectors
        weights = []
        # Vertices
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        # Edge midpoints
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)
        # Center
        w_center = np.ones(n_obj) / n_obj
        weights.append(w_center)
        
        weights = [w / w.sum() for w in weights]

        # Select diverse starting points from initial samples
        # Simple diversity: pick points that are far apart in objective space
        start_points = []
        if len(X_samples) > 0:
            F_arr = np.array(F_samples)
            # Normalize F for distance calculation
            F_min = F_arr.min(axis=0)
            F_max = F_arr.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F_arr - F_min) / F_range
            
            # Greedy diversity selection
            selected_indices = []
            # Start with best point (lowest sum)
            best_idx = np.argmin(np.sum(F_arr, axis=1))
            selected_indices.append(best_idx)
            
            for _ in range(min(5, len(X_samples))):
                if len(selected_indices) >= 5: break
                # Find point farthest from selected set
                max_dist = -1
                next_idx = -1
                for i in range(len(X_samples)):
                    if i in selected_indices: continue
                    min_dist_to_selected = np.min([np.linalg.norm(F_norm[i] - F_norm[sel]) for sel in selected_indices])
                    if min_dist_to_selected > max_dist:
                        max_dist = min_dist_to_selected
                        next_idx = i
                if next_idx != -1:
                    selected_indices.append(next_idx)
            
            start_points = [X_samples[i].copy() for i in selected_indices]
        
        # If we don't have enough start points, add random ones
        while len(start_points) < 5:
            start_points.append(lo + (hi - lo) * rng.random(dim))

        # Define a safe problem wrapper that tracks evaluations
        def safe_problem(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            return f

        # Run Nelder-Mead for each weight and start point combination
        # Limit total runs to ensure we stay within budget
        # Nelder-Mead in 4D uses ~4 evals/iteration. 
        # maxiter=10 -> ~40 evals/run.
        # We have ~170 evals left. 170/40 ~ 4 runs.
        # We will try 1 run per weight (5 weights) but stop if budget runs out.
        
        for w in weights:
            if not time_ok() or eval_count >= budget - 10: # Keep 10 evals buffer
                break
            
            # Pick a start point. Cycle through them or pick random.
            # Let's pick the start point that is "best" for this weight to give NM a head start
            best_start_idx = 0
            best_score = np.inf
            for idx, x_sp in enumerate(start_points):
                # Estimate score at start point
                # We already have F_samples, but start_points might be a subset.
                # Let's just pick a random start point to ensure diversity in search paths
                pass
            
            x0 = start_points[rng.integers(len(start_points))]

            def scalar_obj(x, weights=w):
                f = safe_problem(x)
                return np.dot(weights, f)

            bounds_opt = list(zip(lo, hi))
            
            try:
                res = opt.minimize(
                    scalar_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 15, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if res.success:
                    x_best = np.clip(res.x, lo, hi)
                    # We must re-evaluate to get the exact F for the new point
                    # and to ensure it's recorded in our samples list correctly
                    # Note: safe_problem already incremented eval_count.
                    # We need to add this to X_samples/F_samples.
                    # But wait, safe_problem is used inside scalar_obj.
                    # The final point res.x might not have been evaluated in the last call?
                    # Nelder-Mead usually returns the best point found.
                    # Let's evaluate it explicitly to be sure.
                    f_best = evaluate(x_best)
                    X_samples.append(x_best)
                    F_samples.append(f_best)
            except Exception:
                pass

            # Check time and budget
            if not time_ok():
                break

        # Phase 3: Final Polish
        # Use any remaining "tracked" budget to polish the best points
        if time_ok() and eval_count < budget and F_samples:
            F_arr = np.array(F_samples)
            nondom_idx = get_nondominated_indices(F_arr)
            
            # Polish top 3 non-dominated points
            n_polish = min(3, len(nondom_idx), (budget - eval_count) // 10)
            
            for idx in nondom_idx[:n_polish]:
                if not time_ok() or eval_count >= budget - 5:
                    break
                
                x0 = X_samples[idx].copy()
                
                # Try a few different weights for polishing
                for w in weights[:3]:
                    if not time_ok() or eval_count >= budget - 5:
                        break
                    
                    def polish_obj(x, weights=w):
                        f = safe_problem(x)
                        return np.dot(weights, f)
                    
                    bounds_opt = list(zip(lo, hi))
                    try:
                        res = opt.minimize(
                            polish_obj,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 5, 'xatol': 1e-5, 'fatol': 1e-8}
                        )
                        if res.success:
                            x_best = np.clip(res.x, lo, hi)
                            f_best = evaluate(x_best)
                            X_samples.append(x_best)
                            F_samples.append(f_best)
                    except Exception:
                        pass

        return X_samples
