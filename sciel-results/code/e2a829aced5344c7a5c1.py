"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Weighted-Sum Grid Search and L-BFGS-B refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, f_new, archive_F):
        """Check if f_new is dominated by any point in archive_F."""
        if len(archive_F) == 0:
            return False
        # f_new is dominated if there exists a in archive such that a <= f_new and a != f_new
        # Broadcasting: (1, n_obj) vs (n_archive, n_obj)
        # We want: any( all(a <= f_new, axis=1) and any(a < f_new, axis=1) )
        # Efficient check:
        # If any archive point is <= f_new in all objectives AND < in at least one.
        # Let's just loop for clarity as archive size is small (<= 100-200)
        for a in archive_F:
            if np.all(a <= f_new) and np.any(a < f_new):
                return True
        return False

    def _update_archive(self, f_new, x_new, archive_F, archive_X, budget_used):
        """Update archive with new point if non-dominated. Returns updated archives."""
        if self._is_dominated(f_new, archive_F):
            return archive_F, archive_X

        # Remove dominated points from archive
        new_archive_F = []
        new_archive_X = []
        for i, (a_f, a_x) in enumerate(zip(archive_F, archive_X)):
            # If a_f is dominated by f_new, discard it
            if np.all(f_new <= a_f) and np.any(f_new < a_f):
                continue
            new_archive_F.append(a_f)
            new_archive_X.append(a_x)
        
        new_archive_F.append(f_new)
        new_archive_X.append(x_new)
        
        return new_archive_F, new_archive_X

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        archive_F = []
        archive_X = []

        # Phase 1: Weighted-Sum Grid Search
        # RE34 has 3 objectives. We can use barycentric coordinates for weights.
        # Number of grid points: K^2 / 2 roughly.
        # Let's use K=20 -> ~210 points.
        # Budget 200. So K=14 -> 14*15/2 = 105 points.
        # Let's aim for ~100-120 points in Phase 1 to leave room for refinement.
        K = 14 
        weights = []
        for i in range(K):
            for j in range(K - i):
                k = K - i - j
                w = np.array([i, j, k], dtype=float) / (K - 1)
                # Avoid zero weights if possible, or handle them
                if np.any(w == 0):
                    # Small epsilon to avoid singularity or boundary issues, though L-BFGS handles bounds
                    pass
                weights.append(w)
        
        # Shuffle weights to diversify initial spread? No, systematic is better for WSGS.
        # But random order might help if early convergence happens? 
        # Systematic sweep is the point.
        
        for w in weights:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            def weighted_sum(x):
                try:
                    f = problem(x)
                    return np.dot(f, w)
                except:
                    return 1e9

            # Initial guess: center of box
            x0 = (lo + hi) / 2.0
            
            # Optimize
            res = minimize(weighted_sum, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
            
            if res.success or res.status in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                x_opt = res.x
                f_opt = problem(x_opt)
                used += 1
                
                # Update archive
                archive_F, archive_X = self._update_archive(f_opt, x_opt, archive_F, archive_X, used)
            else:
                # Even if failed, record the last point?
                x_opt = res.x
                f_opt = problem(x_opt)
                used += 1
                archive_F, archive_X = self._update_archive(f_opt, x_opt, archive_F, archive_X, used)

        # Phase 2: Local Refinement
        # Take the current non-dominated archive and try to refine each point
        # using L-BFGS-B with slightly perturbed starting points or tighter bounds?
        # Actually, L-BFGS-B already converged. 
        # Better: Use the archive points as seeds for a local search with smaller steps?
        # Or, run L-BFGS-B again with different initializations?
        
        # A better strategy for Phase 2:
        # The WSGS might have missed some regions if the front is non-convex.
        # RE34 is likely non-convex.
        # Let's do a few random restarts of L-BFGS-B with random weights?
        # Or, just use remaining budget for random local search around archive points.
        
        remaining_budget = budget - used
        if remaining_budget <= 0:
            return np.array(archive_X)

        # Strategy for Phase 2:
        # For each point in the archive, try to improve it by doing a local search
        # with a random perturbation.
        # Or, simply do more WSGS with finer grid?
        # Let's do a hybrid: Random perturbations of archive points + L-BFGS-B polish.
        
        # Create a list of candidates for refinement
        candidates = list(zip(archive_X, archive_F))
        
        # Shuffle candidates to ensure diverse improvement
        np.random.shuffle(candidates)
        
        for x_seed, f_seed in candidates:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Perturb the seed
            # Small Gaussian noise
            scale = 0.1 * (hi - lo)
            x_pert = x_seed + np.random.normal(0, scale, dim)
            x_pert = np.clip(x_pert, lo, hi)
            
            # Which weight vector to use? 
            # Use the one that was closest to this point? 
            # Or just a random weight?
            # Let's use a random weight vector.
            w = np.random.dirichlet([1, 1, 1])
            
            def weighted_sum_local(x):
                try:
                    f = problem(x)
                    return np.dot(f, w)
                except:
                    return 1e9

            res = minimize(weighted_sum_local, x_pert, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30})
            
            if res.success or res.status in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                x_opt = res.x
                f_opt = problem(x_opt)
                used += 1
                archive_F, archive_X = self._update_archive(f_opt, x_opt, archive_F, archive_X, used)
            else:
                x_opt = res.x
                f_opt = problem(x_opt)
                used += 1
                archive_F, archive_X = self._update_archive(f_opt, x_opt, archive_F, archive_X, used)
                
            # If we still have budget, maybe do another round?
            # But we only have ~100-150 points in archive. 
            # If budget is 200, and Phase 1 used ~105, we have ~95 left.
            # We will process the first ~95 candidates.
            
        return np.array(archive_X)
# EVOLVE-BLOCK-END
