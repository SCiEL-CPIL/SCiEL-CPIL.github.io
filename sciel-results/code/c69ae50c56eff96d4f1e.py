import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Optimizer:
    """
    Surrogate-assisted global optimizer using a lightweight GP-like model
    for exploration, combined with L-BFGS-B and Nelder-Mead for exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 40s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        rng = np.random.default_rng(42)

        # --- Phase 1: Diverse initialization ---
        # Start with structured + random points to cover the space
        init_points = [np.zeros(dim)]
        for _ in range(8):
            init_points.append(rng.uniform(-5, 5, dim))
        # Axis-aligned points at key locations
        for i in range(dim):
            for val in (-3, 0, 3):
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)

        unique_starts = []
        for s in init_points:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=0.15) for u in unique_starts):
                unique_starts.append(s)

        # Evaluate initial points
        X_eval = []
        f_eval = []
        for s in unique_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                X_eval.append(s.copy())
                f_eval.append(f)
                update_best(s, f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)
            X_eval.append(best_x.copy())
            f_eval.append(best_f)

        # --- Phase 2: Surrogate-assisted search ---
        # Use a simple distance-based surrogate with adaptive length scales
        # For each candidate, compute a score balancing exploitation and exploration
        
        n_candidates = 50  # number of candidate points to evaluate per iteration
        max_iterations = 8  # number of major iterations
        
        for iteration in range(max_iterations):
            if calls[0] >= budget or time.time() > deadline:
                break
            
            X_arr = np.array(X_eval)
            f_arr = np.array(f_eval)
            
            # Compute distances from existing points to use as a simple kernel
            # Adaptive length scale: shrink as we get more confident
            base_ls = 2.0 * (1.0 - 0.08 * iteration)
            base_ls = max(base_ls, 0.3)
            
            # Generate candidate points: mix of local and global
            candidates = []
            
            # Local candidates around best point
            for _ in range(20):
                local_pt = best_x + rng.standard_normal(dim) * base_ls * 0.5
                local_pt = np.clip(local_pt, -5, 5)
                candidates.append(local_pt)
            
            # Global candidates: random with some structure
            for _ in range(15):
                candidates.append(rng.uniform(-5, 5, dim))
            
            # Candidates near other good points
            if len(f_arr) > 1:
                sorted_idx = np.argsort(f_arr)
                for idx in sorted_idx[:5]:
                    for _ in range(3):
                        near_pt = X_arr[idx] + rng.standard_normal(dim) * base_ls * 0.3
                        near_pt = np.clip(near_pt, -5, 5)
                        candidates.append(near_pt)
            
            # Boundary exploration
            for _ in range(10):
                pt = rng.uniform(-5, 5, dim)
                # Push some coordinates toward boundaries
                for d in range(dim):
                    if rng.random() < 0.3:
                        pt[d] = rng.choice([-4.5, 4.5])
                candidates.append(np.clip(pt, -5, 5))
            
            # Deduplicate candidates
            unique_candidates = []
            for c in candidates:
                if not any(np.allclose(c, u, atol=0.1) for u in unique_candidates):
                    unique_candidates.append(c)
            
            # Score each candidate using a simple acquisition function
            # Balance: predicted improvement (exploitation) + distance to known points (exploration)
            scored_candidates = []
            for c in unique_candidates:
                # Distance to nearest known point (exploration term)
                if len(X_arr) > 0:
                    dists = np.linalg.norm(X_arr - c, axis=1)
                    min_dist = np.min(dists)
                else:
                    min_dist = 5.0
                
                # Simple surrogate prediction: weighted average of nearby points
                # with inverse distance weighting
                if len(X_arr) > 0:
                    dists = np.linalg.norm(X_arr - c, axis=1)
                    # Use inverse distance weighting with adaptive scale
                    weights = np.exp(-dists**2 / (2 * base_ls**2))
                    if np.sum(weights) > 1e-10:
                        pred_f = np.sum(weights * f_arr) / np.sum(weights)
                    else:
                        pred_f = np.mean(f_arr)
                else:
                    pred_f = 0.0
                
                # Acquisition: expected improvement-like score
                # Higher is better for selection
                exploit = best_f - pred_f  # potential improvement
                explore = min_dist / 5.0   # normalized distance
                # Adaptive balance: more exploration early, more exploitation later
                gamma = 0.4 + 0.4 * (iteration / max_iterations)
                score = gamma * explore + (1 - gamma) * (exploit / (abs(best_f) + 1.0))
                
                scored_candidates.append((score, c))
            
            # Select top candidates
            scored_candidates.sort(key=lambda x: -x[0])
            n_select = min(n_candidates, len(scored_candidates), budget - calls[0])
            
            for score, c in scored_candidates[:n_select]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    f = wrapped(c)
                    X_eval.append(c.copy())
                    f_eval.append(f)
                    update_best(c, f)
                except Exception:
                    break
            
            # Local polish on the best point found so far
            if calls[0] < budget and time.time() < deadline:
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 20, 'maxls': 10, 'gtol': 1e-5}
                    )
                    update_best(res.x, res.fun)
                    # Add to our knowledge base
                    X_eval.append(res.x.copy())
                    f_eval.append(res.fun)
                except Exception:
                    pass
        
        # --- Phase 3: Final exploitation ---
        # Use remaining budget for intensive local optimization
        remaining = budget - calls[0]
        if best_x is not None and remaining > 10 and time.time() < deadline:
            # Try multiple local optimizers from best point
            for method, opts in [
                ('L-BFGS-B', {'maxiter': 30, 'maxls': 15, 'gtol': 1e-6}),
                ('Nelder-Mead', {'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
            ]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method=method,
                        bounds=[(-5, 5)] * dim if method == 'L-BFGS-B' else None,
                        options=opts
                    )
                    update_best(res.x, res.fun)
                except Exception:
                    pass
        
        # Final tight polish
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10,
                             'fatol': 1e-14, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        return best_x, best_f
