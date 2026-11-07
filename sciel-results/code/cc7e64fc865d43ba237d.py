"""Multi-objective optimizer for WFG4 using LHS + archive-based local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS exploration + archive-based local search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        span = hi - lo

        ref = np.array([3.0, 5.0, 7.0])

        archive_X = []
        archive_F = []

        def add_to_archive(x, f):
            """Add (x, f) to archive if non-dominated, removing any dominated points."""
            if len(archive_F) > 0:
                af = np.array(archive_F)
                if np.any(np.all(af <= f, axis=1) & np.any(af < f, axis=1)):
                    return
            keep = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    keep.append(i)
            non_local = [(archive_X[i], archive_F[i]) for i in keep]
            non_local.append((x, f))
            archive_X.clear()
            archive_F.clear()
            for px, pf in non_local:
                archive_X.append(px)
                archive_F.append(pf)

        def time_left():
            return deadline - (time.time() - start_time)

        rng = np.random.default_rng(42)
        used = 0

        # Phase 1: Latin Hypercube Sampling for broad exploration
        # Use ~50% of budget for LHS
        lhs_count = min(budget, max(40, budget // 2))
        
        lhs_X = np.zeros((lhs_count, dim))
        for d in range(dim):
            perm = rng.permutation(lhs_count)
            lhs_X[:, d] = (perm + rng.random(lhs_count)) / lhs_count
        
        lhs_X = lo + lhs_X * span
        
        for i in range(lhs_count):
            if time_left() < 5.0 or used >= budget:
                break
            f = problem(lhs_X[i])
            used += 1
            if np.all(f < ref):
                add_to_archive(lhs_X[i].copy(), f.copy())

        if used >= budget or time_left() < 10.0:
            return np.array(archive_X) if archive_X else np.zeros((1, dim))

        # Phase 2: Archive-based local search (coordinate perturbation)
        # For each archive point, try small perturbations in each dimension
        remaining = budget - used
        n_archive = len(archive_X)
        
        if n_archive == 0:
            # No valid points found; do random search
            for _ in range(remaining):
                if time_left() < 5.0 or used >= budget:
                    break
                x = lo + rng.random(dim) * span
                f = problem(x)
                used += 1
                if np.all(f < ref):
                    add_to_archive(x.copy(), f.copy())
            return np.array(archive_X) if archive_X else np.zeros((1, dim))

        # Sort archive by sum of objectives (prioritize better points)
        order = np.argsort(np.sum(archive_F, axis=1))
        
        # Local search: for each archive point, try perturbations
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02]
        
        for idx in order:
            if used >= budget or time_left() < 5.0:
                break
            
            base_x = archive_X[idx]
            base_f = archive_F[idx]
            
            # Try perturbations in each dimension with decreasing step sizes
            for d in range(dim):
                for step in step_sizes:
                    if used >= budget or time_left() < 5.0:
                        break
                    
                    # Try positive and negative perturbations
                    for direction in [1, -1]:
                        if used >= budget or time_left() < 5.0:
                            break
                        
                        trial = base_x.copy()
                        trial[d] = base_x[d] + direction * step * span[d]
                        trial = np.clip(trial, lo, hi)
                        
                        f_trial = problem(trial)
                        used += 1
                        
                        if np.all(f_trial < ref):
                            add_to_archive(trial.copy(), f_trial.copy())
                        
                        # If improved, update base for further search
                        if np.all(f_trial <= base_f) and np.any(f_trial < base_f):
                            base_x = trial
                            base_f = f_trial
                            break  # Move to next dimension

        return np.array(archive_X) if archive_X else np.zeros((1, dim))
# EVOLVE-BLOCK-END
