import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = [0]  # Mutable counter for nested functions
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            f = np.array(f)
            # Check if dominated by any existing point in archive
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            return True

        def safe_eval(x):
            if used[0] >= self.budget:
                return None
            try:
                f = problem(x)
                used[0] += 1
                return f
            except Exception:
                return None

        # Phase 1: Latin Hypercube Sampling for initial exploration
        # Use 10-15% of budget for initialization
        n_init = max(10, min(20, self.budget // 10))
        
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * (hi - lo)
        
        for x in X_init:
            if time.time() > deadline or used[0] >= self.budget:
                break
            f = safe_eval(x)
            if f is not None:
                add_to_archive(x, f)
        
        # Phase 2: Weighted Sum Optimization with L-BFGS-B
        # Generate a diverse set of weight vectors on the simplex
        # Vertices
        weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            # Edge midpoints
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            # Center
            np.array([1.0/3, 1.0/3, 1.0/3]),
            # Additional random weights for coverage
        ]
        
        # Add random weights
        n_random_weights = 15
        for _ in range(n_random_weights):
            w = np.random.random(3)
            w = w / np.sum(w)
            weights.append(w)
        
        weights = np.array(weights)
        
        # Allocate budget for local optimization
        # We have used ~n_init evals. Remaining: budget - used
        # Let's say we want to do ~5-10 evals per weight direction
        # Total weights: ~22
        # If we use 10 evals each, that's 220 evals.
        # Budget is 200. So we need to be careful.
        # Let's dynamically determine maxfev based on remaining budget.
        
        remaining_budget = self.budget - used[0]
        n_weights = len(weights)
        
        # Ensure we don't exceed budget
        # Each minimize call might use up to maxfev evaluations.
        # Let's set maxfev to be small enough that n_weights * maxfev <= remaining_budget
        # But we also want enough iterations.
        # Let's try maxfev = 8. 22 * 8 = 176. Plus 20 init = 196. Safe.
        maxfev_per_run = 8
        
        # If remaining budget is very small, reduce maxfev
        if n_weights * maxfev_per_run > remaining_budget:
            maxfev_per_run = max(1, remaining_budget // n_weights)
            
        # Sort weights by priority? Maybe vertices first?
        # The list is already structured.
        
        for w in weights:
            if time.time() > deadline or used[0] >= self.budget:
                break
            
            # Start from a point in the archive that is "good" for this weight
            # or just the center of the box.
            # Using a random point from the archive can help escape local minima.
            if archive_X:
                # Pick a random point from the archive
                idx = np.random.randint(len(archive_X))
                x0 = archive_X[idx].copy()
            else:
                x0 = (lo + hi) / 2.0
            
            def weighted_obj(x, w=w):
                if used[0] >= self.budget:
                    return 1e10
                f = problem(x)
                used[0] += 1
                return np.dot(w, f)
            
            try:
                res = minimize(
                    weighted_obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 5, 'maxfev': maxfev_per_run, 'ftol': 1e-4}
                )
                
                # Evaluate the best point found
                if used[0] < self.budget:
                    x_best = res.x
                    f_best = safe_eval(x_best)
                    if f_best is not None:
                        add_to_archive(x_best, f_best)
                        
            except Exception:
                pass

        # Phase 3: Final Polish on the best points
        # If we have budget left, do a quick local search on the top points
        # Sort archive by sum of objectives (simple heuristic for "goodness")
        if archive_F and used[0] < self.budget:
            archive_F_arr = np.array(archive_F)
            sums = np.sum(archive_F_arr, axis=1)
            sorted_idx = np.argsort(sums)
            
            # Pick top 5
            top_indices = sorted_idx[:5]
            
            for idx in top_indices:
                if time.time() > deadline or used[0] >= self.budget:
                    break
                
                x0 = archive_X[idx].copy()
                
                def sum_obj(x):
                    if used[0] >= self.budget:
                        return 1e10
                    f = problem(x)
                    used[0] += 1
                    return np.sum(f)
                
                try:
                    res = minimize(
                        sum_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 3, 'maxfev': 5, 'ftol': 1e-5}
                    )
                    if used[0] < self.budget:
                        x_best = res.x
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_to_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X
