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
        deadline = start_time + 285  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = [0]
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            f = np.array(f)
            # Check if dominated by any existing point in archive
            for af in archive_F:
                if dominates(af, f):
                    return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
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

        def time_check():
            return time.time() < deadline and used[0] < self.budget

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = 20
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * (hi - lo)
        
        for x in X_init:
            if not time_check():
                break
            f = safe_eval(x)
            if f is not None:
                add_to_archive(x, f)
        
        # Phase 2: Weighted Sum Optimization with multiple weight vectors
        # Use a systematic set of weight vectors for better coverage
        weights = []
        
        # Vertices
        weights.append(np.array([1.0, 0.0, 0.0]))
        weights.append(np.array([0.0, 1.0, 0.0]))
        weights.append(np.array([0.0, 0.0, 1.0]))
        
        # Edge midpoints
        weights.append(np.array([0.5, 0.5, 0.0]))
        weights.append(np.array([0.5, 0.0, 0.5]))
        weights.append(np.array([0.0, 0.5, 0.5]))
        
        # Center
        weights.append(np.array([1.0/3, 1.0/3, 1.0/3]))
        
        # Additional structured weights for better coverage
        for w1 in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            for w2 in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    weights.append(np.array([w1, w2, w3]))
        
        # Remove duplicates and normalize
        unique_weights = []
        for w in weights:
            w_norm = w / np.sum(w)
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w_norm, uw, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w_norm)
        
        weights = unique_weights
        
        # Allocate budget for local optimization
        remaining_budget = self.budget - used[0]
        n_weights = len(weights)
        
        # Calculate maxfev per run
        if n_weights > 0:
            maxfev_per_run = max(1, remaining_budget // n_weights)
        else:
            maxfev_per_run = 1
        
        # Limit maxfev to reasonable range
        maxfev_per_run = min(maxfev_per_run, 20)
        
        # Sort weights: vertices first, then edges, then center, then others
        def weight_priority(w):
            # Vertices have one component = 1, others = 0
            if np.any(np.abs(w - 1.0) < 1e-6) and np.sum(np.abs(w) < 1e-6) == 2:
                return 0
            # Edge midpoints have two components = 0.5
            if np.sum(np.abs(w - 0.5) < 1e-6) == 2 and np.sum(np.abs(w) < 1e-6) == 1:
                return 1
            # Center has all components equal
            if np.allclose(w, w[0], atol=1e-6):
                return 2
            # Others
            return 3
        
        weights.sort(key=weight_priority)
        
        for w in weights:
            if not time_check():
                break
            
            # Start from the center of the box
            x0 = (lo + hi) / 2.0
            
            def weighted_obj(x, w=w):
                if used[0] >= self.budget:
                    return 1e10
                try:
                    f = problem(x)
                    used[0] += 1
                    return np.dot(w, f)
                except Exception:
                    return 1e10
            
            try:
                res = minimize(
                    weighted_obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 15, 'maxfev': maxfev_per_run, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                # Evaluate the best point found
                if used[0] < self.budget:
                    x_best = res.x
                    f_best = safe_eval(x_best)
                    if f_best is not None:
                        add_to_archive(x_best, f_best)
                        
            except Exception:
                pass
        
        # Phase 3: Polish archive points with multiple objectives
        if archive_F and used[0] < self.budget:
            archive_F_arr = np.array(archive_F)
            n_archive = len(archive_F)
            
            # Polish each archive point with sum and weighted objectives
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not time_check():
                    break
                
                x0 = ax.copy()
                
                # Use sum of objectives as polishing objective
                def sum_obj(x):
                    if used[0] >= self.budget:
                        return 1e10
                    try:
                        f = problem(x)
                        used[0] += 1
                        return np.sum(f)
                    except Exception:
                        return 1e10
                
                try:
                    res = minimize(
                        sum_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 8, 'maxfev': 4, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                    
                    if used[0] < self.budget:
                        x_best = res.x
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_to_archive(x_best, f_best)
                            
                except Exception:
                    pass
        
        return archive_X
