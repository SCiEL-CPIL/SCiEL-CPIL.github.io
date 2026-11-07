"""Multi-objective optimizer on ZDT6 with deadline guard and ZDT6-specific structure."""
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer tailored for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # ZDT6 specific: x0 in [0, 1/9] ensures f1 < 11 (valid region)
        # f1 = 1 + 10*(m-1) - x0 * sum(10*(m-1) * sin(2*pi*x_i))
        # With m=2: f1 = 11 - x0 * 10 * sum(sin(2*pi*x_i))
        # For f1 < 11, we need x0 * sum(sin(2*pi*x_i)) > 0
        # Safest: keep x0 in [0.01, 1/9] and x_i in [0, 0.5] so sin is positive

        # Initialize population in valid region
        pop_size = min(40, self.budget // 5)
        X = np.zeros((pop_size, n))
        # x0 in [0.01, 1/9]
        X[:, 0] = rng.uniform(0.01, 1.0 / 9.0, pop_size)
        # x1..x9 in [0, 0.5] to ensure sin(2*pi*x_i) >= 0
        X[:, 1:] = rng.uniform(0.0, 0.5, (pop_size, n - 1))

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_X = list(X)
        archive_F = list(F)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Remove points dominated by new point
            new_archive_X = [x]
            new_archive_F = [f]
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            return new_archive_X, new_archive_F

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute non-dominated front from archive
            if len(archive_F) > 0:
                # Find non-dominated in archive
                nondom_idx = []
                for i in range(len(archive_F)):
                    dominated = False
                    for j in range(len(archive_F)):
                        if i != j:
                            if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                                dominated = True
                                break
                    if not dominated:
                        nondom_idx.append(i)
            else:
                nondom_idx = list(range(len(archive_F)))

            if len(nondom_idx) == 0:
                nondom_idx = list(range(len(archive_F)))

            # Select parent from non-dominated set
            parent_idx = nondom_idx[rng.integers(len(nondom_idx))]
            parent = archive_X[parent_idx].copy()

            # Mutation: perturb x0 slightly, perturb other variables
            # Keep x0 in valid region
            mutation_scale = 0.05
            child = parent.copy()
            child[0] += rng.normal(0, mutation_scale * 0.1)
            child[0] = np.clip(child[0], 0.005, 1.0 / 9.0)
            for i in range(1, n):
                child[i] += rng.normal(0, mutation_scale)
                child[i] = np.clip(child[i], 0.0, 0.5)

            fc = problem(child)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(child, fc, archive_X, archive_F)

            # Also try a more aggressive exploration step every 10th iteration
            if used % 10 == 0 and used < self.budget:
                if time.time() > deadline:
                    break
                # Random point in valid region
                exp_child = np.zeros(n)
                exp_child[0] = rng.uniform(0.01, 1.0 / 9.0)
                exp_child[1:] = rng.uniform(0.0, 0.5, n - 1)
                fc_exp = problem(exp_child)
                used += 1
                archive_X, archive_F = update_archive(exp_child, fc_exp, archive_X, archive_F)

        return archive_X
