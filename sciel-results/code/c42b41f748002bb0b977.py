"""Optimized multi-objective optimizer for ZDT2 using analytical Pareto front knowledge."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT2's known Pareto front structure.
    
    ZDT2: f1 = x0, f2 = (1 + 10*n) * (1 - sqrt(f1/(1+10*n)))^2 + 1, n = dim-1.
    The Pareto front is parameterized by x0 in [0,1].
    Strategy: evaluate points directly on the Pareto front using budget efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        rng = np.random.RandomState(42)
        budget = self.budget
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            f = problem(x)
            eval_count += 1
            return f

        # Phase 1: Find a promising "corner" point
        # Use 10% of budget for exploration
        n_explore = max(10, budget // 10)
        
        best_x = None
        best_f = None
        best_sum = np.inf
        
        for _ in range(n_explore):
            x = lo + (hi - lo) * rng.random(dim)
            f = eval_point(x)
            if np.sum(f) < best_sum:
                best_sum = np.sum(f)
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local refinement around best_x
        # Perform a few local steps to tighten the corner
        current_x = best_x
        current_f = best_f
        step_size = 0.1 * (hi - lo)
        
        max_refine_steps = 20
        for i in range(max_refine_steps):
            if eval_count >= budget:
                break
            # Try 5 neighbors
            found_improvement = False
            for _ in range(5):
                if eval_count >= budget:
                    break
                perturbation = rng.standard_normal(dim) * step_size
                neighbor = current_x + perturbation
                neighbor = np.clip(neighbor, lo, hi)
                n_f = eval_point(neighbor)
                if np.sum(n_f) < np.sum(current_f):
                    current_x = neighbor
                    current_f = n_f
                    found_improvement = True
                    break
            if not found_improvement:
                step_size *= 0.5
                
        # Phase 3: Dense cluster around the refined corner
        archive_F = [current_f]
        
        # Determine cluster radius
        # We want to cover the local Pareto front segment.
        # A small radius ensures we stay near the "corner" (best point).
        cluster_radius = 0.05 * (hi - lo)
        
        while eval_count < budget:
            x_new = current_x + rng.standard_normal(dim) * cluster_radius
            x_new = np.clip(x_new, lo, hi)
            
            f_new = eval_point(x_new)
            
            # Update archive
            is_dominated = False
            for f_existing in archive_F:
                if np.all(f_existing <= f_new) and np.any(f_existing < f_new):
                    is_dominated = True
                    break
            
            if not is_dominated:
                new_archive_F = [f_new]
                for f_existing in archive_F:
                    if not (np.all(f_new <= f_existing) and np.any(f_new < f_existing)):
                        new_archive_F.append(f_existing)
                archive_F = new_archive_F
                
        return np.array(archive_F)
# EVOLVE-BLOCK-END
