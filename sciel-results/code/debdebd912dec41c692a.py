"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven archive-based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random

        # Archive: list of (x, f) for non-dominated points
        archive_X = []
        archive_F = []
        used = 0

        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive_F."""
            for f in archive_F:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            """Add point to archive if non-dominated. Returns True if added."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any archive point
            for i, f_arch in enumerate(archive_F):
                if dominates(f_arch, f):
                    return False
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i, (x_arch, f_arch) in enumerate(zip(archive_X, archive_F)):
                if not dominates(f, f_arch):
                    new_archive_X.append(x_arch)
                    new_archive_F.append(f_arch)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        def evaluate_point(x):
            """Evaluate a point and add to archive if non-dominated."""
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            used += 1
            added = add_to_archive(x, f)
            return added

        # Phase 1: Diverse initialization
        # Use Latin Hypercube Sampling for better coverage
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            # LHS: divide each dimension into n_init equal parts, pick one per part
            x = np.zeros(dim)
            for d in range(dim):
                part = (i + rng.random()) / n_init
                x[d] = lo[d] + (hi[d] - lo[d]) * part
            x = np.clip(x, lo, hi)
            evaluate_point(x)

        # Phase 2: Archive-guided search
        # Strategies:
        # 1. Interpolation between archive points
        # 2. Mutation of archive points
        # 3. Random restarts in promising regions

        while used < budget:
            if time.time() > deadline:
                break

            # Choose a strategy based on archive size
            if len(archive_X) < 2:
                # Need more diversity
                x = lo + (hi - lo) * rng.random(dim)
                evaluate_point(x)
                continue

            # Strategy selection
            r = rng.random()

            if r < 0.4 and len(archive_X) >= 2:
                # Interpolation between two archive points
                i1, i2 = rng.choice(len(archive_X), 2, replace=False)
                alpha = rng.random()
                x = alpha * archive_X[i1] + (1 - alpha) * archive_X[i2]
                x = np.clip(x, lo, hi)
                evaluate_point(x)

            elif r < 0.7:
                # Mutation of a random archive point
                i = rng.randint(len(archive_X))
                # Adaptive step: smaller as archive grows
                step_size = 0.1 * (1.0 / (1.0 + len(archive_X) / 10.0))
                x = archive_X[i] + step_size * (hi - lo) * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                evaluate_point(x)

            elif r < 0.85:
                # Differential mutation: x = x1 + 0.5*(x2 - x3)
                if len(archive_X) >= 3:
                    i1, i2, i3 = rng.choice(len(archive_X), 3, replace=False)
                    x = archive_X[i1] + 0.5 * (archive_X[i2] - archive_X[i3])
                    x = np.clip(x, lo, hi)
                    evaluate_point(x)
                else:
                    x = lo + (hi - lo) * rng.random(dim)
                    evaluate_point(x)

            else:
                # Local search around best hypervolume contributor
                # Approximate: pick a random archive point and do small perturbation
                i = rng.randint(len(archive_X))
                step_size = 0.05 * (hi - lo)
                x = archive_X[i] + step_size * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                evaluate_point(x)

        return archive_X
# EVOLVE-BLOCK-END
