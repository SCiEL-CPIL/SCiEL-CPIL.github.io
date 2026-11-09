"""Rewritten program for ZDT6 optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using archive-based search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 10s before the 300s limit
        deadline = 300 - 10
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive stores (x, f) pairs
        archive = []
        used = 0
        
        # Helper to check if f_new is dominated by any point in archive_f
        def is_dominated(f_new, archive_f_list):
            for f_a in archive_f_list:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        # Helper to check if f_new dominates f_old
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Strategy: Mix of global exploration and local exploitation
            # ZDT6 Pareto front is primarily determined by x1.
            # x2..x10 should be 0 to minimize g.
            
            if len(archive) == 0 or np.random.rand() < 0.4:
                # Global Exploration
                # Sample x1 uniformly. The function f1(x1) is non-monotonic.
                # Uniform sampling in x1 ensures we hit the oscillations.
                x = np.zeros(self.dim)
                x[0] = np.random.uniform(lo[0], hi[0])
                # Set other variables to lower bound (0)
                # This minimizes g, placing the point on the Pareto front
                for i in range(1, self.dim):
                    x[i] = lo[i]
            else:
                # Local Exploitation
                # Mutate a random point from the current archive
                idx = np.random.randint(len(archive))
                x_parent = archive[idx][0].copy()
                
                # Perturb x1 with Gaussian noise
                # Scale 0.05 allows exploring nearby regions of the non-monotonic curve
                x[0] = x_parent[0] + np.random.normal(0, 0.05)
                
                # Perturb other variables slightly
                # This helps if the optimal g is not exactly 1 (though it is for ZDT6)
                # or to escape numerical plateaus
                for i in range(1, self.dim):
                    x[i] = x_parent[i] + np.random.normal(0, 0.01)
                
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            
            # Update Archive
            # Check if the new point is dominated by any existing point
            dominated = False
            for ax, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove any archive points dominated by the new point
                new_archive = []
                for ax, af in archive:
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_archive.append((ax, af))
                new_archive.append((x, f))
                archive = new_archive

        return archive
# EVOLVE-BLOCK-END
