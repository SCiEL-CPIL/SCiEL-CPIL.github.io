"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start local search optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget
        
        # Archive to store non-dominated points
        # Each entry is (x, f)
        archive = []
        
        def is_dominated(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            # Check if x is dominated by any in archive
            for ax, af in archive:
                if is_dominated(af, f):
                    return False
            # Remove any points in archive dominated by x
            new_archive = []
            for ax, af in archive:
                if not is_dominated(f, af):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            # Limit archive size to 50 to keep memory and computation low
            if len(new_archive) > 50:
                # Simple heuristic: keep the points with the smallest sum of objectives
                # This biases towards the "knee" but ensures we don't overflow
                new_archive.sort(key=lambda item: np.sum(item[1]))
                new_archive = new_archive[:50]
            archive[:] = new_archive
            return True

        evaluations = 0
        
        # Phase 1: Intelligent Random Initialization
        # ZDT4 structure: x1 is the "shape" variable, x2..xn should be near 0.
        n_init = 50
        for _ in range(n_init):
            x = np.empty(n)
            # x1 uniform
            x[0] = np.random.uniform(lo[0], hi[0])
            # x2..xn Gaussian centered at 0, std 0.5
            for i in range(1, n):
                x[i] = np.random.normal(0, 0.5)
                x[i] = np.clip(x[i], lo[i], hi[i])
            
            f = problem(x)
            evaluations += 1
            add_to_archive(x, f)
            
        # Phase 2: Local Search
        # Use remaining budget to perturb points in the archive
        while evaluations < budget and archive:
            # Pick a random point from archive to explore
            idx = np.random.randint(0, len(archive))
            x_parent = archive[idx][0]
            
            # Perturb
            # Use a small step size
            step = 0.1
            x_child = x_parent + np.random.normal(0, step, size=n)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            evaluations += 1
            
            # Try to add to archive
            add_to_archive(x_child, f_child)
            
            # If archive is empty (shouldn't happen), break
            if not archive:
                break

        return None

# EVOLVE-BLOCK-END
