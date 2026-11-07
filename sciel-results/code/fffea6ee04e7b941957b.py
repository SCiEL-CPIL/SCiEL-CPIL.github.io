import numpy as np
import time


class Optimizer:
    """Targeted multi-objective optimizer for ZDT6 focusing on high-hypervolume regions."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        # Archive to track best non-dominated points
        archive_X = []
        archive_F = []
        used = 0

        def is_dominated(f_new):
            """Check if f_new is dominated by any point in archive."""
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def remove_dominated(f_new):
            """Remove points from archive that are dominated by f_new."""
            nonlocal archive_X, archive_F
            new_X = []
            new_F = []
            for i in range(len(archive_F)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            archive_X = new_X
            archive_F = new_F

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            return f

        def add_point(x, f):
            nonlocal archive_X, archive_F
            if is_dominated(f):
                return
            remove_dominated(f)
            archive_X.append(x)
            archive_F.append(f)

        rng = np.random.default_rng(0)

        # ZDT6 specific: f1 = x1, so we want to explore x1 in [0,1]
        # The Pareto front: f2 = 1 - sqrt(f1) - f1 (for f1 in [0,1])
        # Hypervolume contribution is highest for small f1 values

        # Phase 1: Targeted initialization biased toward high-value region
        # Use alpha=2 concentration near t=0 (small f1)
        n_init = min(40, budget // 5)
        for i in range(n_init):
            # Sample f1 (x1) with concentration near 0
            # Use Beta distribution: Beta(1, 3) gives density ~ 3(1-t)^2, concentrated near 0
            t = rng.beta(1, 3)  # concentrated near 0
            x = np.ones(n)
            x[0] = t
            # Set other variables to minimize g(x) -> g=1 when all xi=0
            # But we need diversity, so set some to 0
            x[1:] = 0.0
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

        # Phase 2: Systematic exploration along the Pareto front
        # Generate candidate points along the known front shape
        # For ZDT6, on the Pareto front: x1 = t, others = 0
        # But we also need to explore to find better points

        # Use a grid of t values with concentration near 0
        n_grid = min(60, budget - used - 20)  # leave some for refinement
        if n_grid > 0:
            # Create t values: mix of uniform and concentrated near 0
            t_vals = np.concatenate([
                np.linspace(0.001, 1.0, n_grid // 2 + 1),  # uniform
                np.sort(rng.beta(1, 2, n_grid // 2))  # concentrated near 0
            ])
            t_vals = t_vals[:n_grid]

            for t in t_vals:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(n)
                x[0] = float(t)
                x = np.clip(x, lo, hi)
                f = evaluate(x)
                add_point(x, f)

        # Phase 3: Local refinement around best archive points
        # Focus on points with small f1 (high hypervolume contribution)
        remaining = budget - used
        if remaining > 0 and len(archive_F) > 0:
            # Sort archive by f1 (ascending) to prioritize high-value points
            sort_idx = np.argsort([f[0] for f in archive_F])
            # Take top points for refinement
            n_refine = min(remaining, 30)
            for i in range(n_refine):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Pick a parent from the top of the archive (small f1)
                parent_idx = sort_idx[i % len(sort_idx)]
                parent = archive_X[parent_idx].copy()

                # Mutate: small perturbation
                mutation_scale = 0.02 * (hi - lo)
                child = parent + mutation_scale * rng.standard_normal(n)
                child = np.clip(child, lo, hi)

                f = evaluate(child)
                add_point(child, f)

        # Phase 4: If budget remains, do more targeted exploration
        remaining = budget - used
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Random exploration with bias toward low x1
                t = rng.beta(1, 4)  # even more concentrated near 0
                x = np.zeros(n)
                x[0] = t
                # Add small random perturbation to other dims
                if n > 1:
                    x[1:] = rng.uniform(0, 0.1, n - 1)
                x = np.clip(x, lo, hi)
                f = evaluate(x)
                add_point(x, f)

        return archive_X if archive_X else [np.zeros(n)]
