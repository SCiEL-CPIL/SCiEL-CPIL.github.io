import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f1, f2):
        """Check if f1 is dominated by f2 (f2 is better than f1 in at least one objective and no worse in all others)"""
        return np.all(f2 <= f1) and np.any(f2 < f1)

    def _add_to_archive(self, x, f, archive_X, archive_F):
        """Add point to archive if non-dominated, remove dominated points"""
        # Check if new point is dominated by any archive point
        for j in range(len(archive_F)):
            if self._is_dominated(f, archive_F[j]):
                return archive_X, archive_F
        
        # Remove points dominated by new point
        new_X = []
        new_F = []
        for j in range(len(archive_X)):
            if not self._is_dominated(archive_F[j], f):
                new_X.append(archive_X[j])
                new_F.append(archive_F[j])
        
        new_X.append(x)
        new_F.append(f)
        
        return np.array(new_X), np.array(new_F)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        evals = 0
        
        def check_time():
            return time.time() < deadline
        
        # Phase 1: Deterministic LHS initialization
        # Use 40 points for initial spread
        N_lhs = 40
        X_lhs = np.zeros((N_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(N_lhs)
            X_lhs[:, d] = (perm + 0.5) / N_lhs
            X_lhs[:, d] = lo[d] + X_lhs[:, d] * (hi[d] - lo[d])
        
        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])
        evals += N_lhs
        
        # Build initial archive
        archive_X = []
        archive_F = []
        for i in range(N_lhs):
            if not check_time():
                break
            archive_X, archive_F = self._add_to_archive(X_lhs[i], F_lhs[i], 
                                                          np.array(archive_X) if archive_X else np.array([]).reshape(0, dim),
                                                          np.array(archive_F) if archive_F else np.array([]).reshape(0, n_obj))
        
        # If archive is empty, use all points
        if len(archive_X) == 0:
            archive_X = X_lhs.copy()
            archive_F = F_lhs.copy()
        
        # Phase 2: Nelder-Mead refinement on non-dominated points
        # Sort archive by a simple metric (sum of objectives) to prioritize promising points
        if len(archive_F) > 0:
            sums = np.sum(archive_F, axis=1)
            sorted_idx = np.argsort(sums)
            # Only refine the best 15 non-dominated points
            n_refine = min(15, len(archive_X))
            refine_indices = sorted_idx[:n_refine]
        else:
            refine_indices = []
        
        # Allocate remaining budget among refinement points
        remaining_budget = budget - evals
        if remaining_budget > 0 and len(refine_indices) > 0 and check_time():
            # Each Nelder-Mead run uses ~10-20 evaluations, so limit to a few runs
            max_nelder_runs = min(len(refine_indices), max(1, remaining_budget // 25))
            
            for idx in refine_indices[:max_nelder_runs]:
                if not check_time() or evals >= budget:
                    break
                
                x0 = archive_X[idx].copy()
                
                # Use Nelder-Mead to minimize a weighted sum (scalarization)
                # Try different weights to explore different parts of the Pareto front
                weights = np.random.dirichlet(np.ones(n_obj))
                
                def objective(x):
                    nonlocal evals
                    fx = problem(x)
                    evals += 1
                    return np.dot(weights, fx)
                
                # Nelder-Mead settings
                options = {
                    'maxiter': 20,
                    'maxfev': 20,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'disp': False
                }
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     bounds=list(zip(lo, hi)), options=options)
                    if result.success or evals < budget:
                        x_best = result.x
                        f_best = problem(x_best)
                        evals += 1
                        
                        # Add to archive
                        archive_X, archive_F = self._add_to_archive(x_best, f_best,
                                                                      archive_X, archive_F)
                except Exception:
                    pass
        
        # Phase 3: Use remaining budget for additional random perturbations
        while evals < budget and check_time() and len(archive_X) > 0:
            # Pick a random point from the archive
            idx = np.random.randint(len(archive_X))
            x_curr = archive_X[idx].copy()
            
            # Random perturbation with adaptive step size
            step_scale = 0.05 * (hi - lo)
            x_new = x_curr + np.random.normal(0, 1, dim) * step_scale
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            evals += 1
            
            # Add to archive if non-dominated
            archive_X, archive_F = self._add_to_archive(x_new, f_new,
                                                         archive_X, archive_F)
        
        # Return the archive (all recorded non-dominated points)
        if len(archive_X) == 0:
            # Fallback: return the LHS points
            return X_lhs
        else:
            return archive_X
