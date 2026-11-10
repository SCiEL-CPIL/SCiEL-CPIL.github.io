"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using LHS Exploration and Local Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        archive_X = []
        archive_F = []
        
        def is_dominated(f_new, f_set):
            """Check if f_new is dominated by any point in f_set."""
            for f in f_set:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def update_archive(x, f):
            """Add point to archive if non-dominated, and remove dominated points."""
            if is_dominated(f, archive_F):
                return False
            
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                # If existing point i is dominated by new f, skip it
                if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                    continue
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            # Update archive
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        # Phase 1: Global Exploration using Latin Hypercube Sampling
        n_explore = max(10, self.budget // 5)
        
        # Generate LHS points
        lhs_points = np.zeros((n_explore, n))
        for j in range(n):
            idx = np.arange(n_explore)
            perm = rng.permutation(n_explore)
            lhs_points[:, j] = (perm + rng.random(n_explore)) / n_explore
        
        # Map to bounds
        lhs_points = lo + (hi - lo) * lhs_points
        
        for i in range(n_explore):
            x = lhs_points[i]
            f = np.array(problem(x))
            update_archive(x, f)
            
        # Phase 2: Local Refinement
        remaining_budget = self.budget - n_explore
        
        # Adaptive step size
        step_size = 0.1
        
        for _ in range(remaining_budget):
            if not archive_X:
                x = lo + (hi - lo) * rng.random(n)
            else:
                # Pick a random point from the archive
                idx = rng.integers(len(archive_X))
                x_center = archive_X[idx]
                
                # Generate a neighbor
                noise = rng.normal(0, step_size, n) * (hi - lo)
                x = x_center + noise
                x = np.clip(x, lo, hi)
            
            f = np.array(problem(x))
            update_archive(x, f)
            
            # Decay step size
            step_size *= 0.995
            if step_size < 0.001:
                step_size = 0.001

        return archive_X
# EVOLVE-BLOCK-END
