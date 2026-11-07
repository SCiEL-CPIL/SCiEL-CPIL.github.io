# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-like multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size: use a reasonable fraction of budget
        pop_size = min(50, self.budget // 4)
        if pop_size < 10:
            pop_size = max(10, self.budget // 2)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated solutions
        archive_X = X.copy()
        archive_F = F.copy()

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            return np.any(np.all(archive_F <= f, axis=1) & np.any(archive_F < f, axis=1))

        def update_archive(x, f, archive_X, archive_F):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Remove points dominated by f
            keep = ~np.any(np.all(f <= archive_F, axis=1) & np.any(f < archive_F, axis=1), axis=0)
            archive_X = np.vstack([archive_X[keep], x])
            archive_F = np.vstack([archive_F[keep], f])
            return archive_X, archive_F

        # Main evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Select parents from archive (non-dominated solutions)
            if len(archive_X) > 0:
                parent_idx = rng.integers(len(archive_X))
                parent = archive_X[parent_idx]
            else:
                parent = X[rng.integers(len(X))]

            # Mutation: small perturbation
            mutation_scale = 0.05
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update population (replace worst)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx], F[worst_idx] = child, fc

            # Update archive
            archive_X, archive_F = update_archive(child, fc, archive_X, archive_F)

            # Periodically refresh population from archive to maintain diversity
            if used % pop_size == 0 and len(archive_X) >= pop_size:
                # Sample from archive to maintain diversity
                indices = rng.choice(len(archive_X), size=pop_size, replace=False)
                X = archive_X[indices].copy()
                F = archive_F[indices].copy()

        return X
# EVOLVE-BLOCK-END
