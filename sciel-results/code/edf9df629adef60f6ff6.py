"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using LHS sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Latin Hypercube Sampling for initial coverage
        n_lhs = min(40, budget // 5)  # Use 40 points for initial sampling
        lhs_points = []
        for i in range(n_lhs):
            # For each dimension, use a different permutation
            x = np.zeros(dim)
            for d in range(dim):
                # Create a permutation for this dimension
                perm = np.arange(n_lhs)
                rng.shuffle(perm)
                # The i-th point in dimension d gets the i-th element of the permuted range
                x[d] = lo[d] + span[d] * (perm[i] + 0.5) / n_lhs
            lhs_points.append(x)
        
        lhs_points = np.array(lhs_points)
        lhs_f = np.array([problem(x) for x in lhs_points])
        used = n_lhs
        
        # Keep track of all evaluated points
        all_X = lhs_points.copy()
        all_F = lhs_f.copy()
        
        # Phase 2: Local refinement around promising points
        # Sort by sum of objectives (simple scalarization)
        sums = np.sum(all_F, axis=1)
        sorted_indices = np.argsort(sums)
        
        # Try to refine the best 10 points
        n_refine = min(10, len(sorted_indices))
        
        for idx in sorted_indices[:n_refine]:
            if used >= budget or time.time() > deadline - 30:
                break
            
            x0 = all_X[idx]
            f0 = all_F[idx]
            
            # Try Nelder-Mead on a scalarized objective
            # Use weighted sum with different weights to explore different parts of Pareto front
            for w_seed in range(3):
                if used >= budget or time.time() > deadline - 30:
                    break
                
                # Generate random weights
                w = rng.random(self.n_obj)
                w = w / np.sum(w)
                
                def scalar_obj(x):
                    fx = problem(x)
                    return np.dot(fx, w)
                
                # Try local optimization
                try:
                    result = minimize(scalar_obj, x0, method='Nelder-Mead',
                                     options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8})
                    if result.success or result.fun < np.dot(f0, w):
                        x_opt = np.clip(result.x, lo, hi)
                        f_opt = problem(x_opt)
                        used += 1
                        
                        # Check if it's non-dominated
                        if self._is_nondominated(f_opt, all_F):
                            all_X = np.vstack([all_X, [x_opt]])
                            all_F = np.vstack([all_F, [f_opt]])
                except Exception:
                    pass
        
        # Phase 3: Use remaining budget for random exploration if time allows
        if used < budget and time.time() < deadline - 10:
            while used < budget:
                if time.time() > deadline - 5:
                    break
                x_rand = lo + span * rng.random(dim)
                f_rand = problem(x_rand)
                used += 1
                all_X = np.vstack([all_X, [x_rand]])
                all_F = np.vstack([all_F, [f_rand]])
        
        return all_X

    def _is_nondominated(self, f_new, F_archive):
        """Check if f_new is non-dominated by any point in F_archive."""
        if len(F_archive) == 0:
            return True
        # f_new is dominated if exists i: F_archive[i] <= f_new and F_archive[i] != f_new
        dominated = np.any(np.all(F_archive <= f_new, axis=1) & np.any(F_archive < f_new, axis=1))
        return not dominated
# EVOLVE-BLOCK-END
