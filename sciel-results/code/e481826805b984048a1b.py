"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using structured weight grid + Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        used = 0

        # Archive management
        archive_X = []
        archive_F = []

        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            # Add new point if not dominated by existing
            if not is_dominated(f, new_F):
                new_X.append(x)
                new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        def scalarize(weights, x):
            """Weighted sum scalarization."""
            return float(np.dot(weights, problem(x)))

        def nm_optimize(weights, x0, maxiter=20):
            """Nelder-Mead local optimization for a given weight vector."""
            nonlocal used
            if used >= self.budget or time.time() > deadline - 10:
                return x0, None

            # Use Nelder-Mead on the scalarized objective
            def objective(x):
                nonlocal used
                if used >= self.budget or time.time() > deadline - 5:
                    return np.inf
                used += 1
                return float(np.dot(weights, problem(x)))

            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
                if used < self.budget and time.time() < deadline - 5:
                    f = np.array(problem(res.x))
                    update_archive(res.x, f)
                return res.x, f if used <= self.budget else None
            except Exception:
                return x0, None

        # Phase 1: Structured weight grid for diverse exploration
        # Use a grid of weight vectors in 3D simplex
        n_grid = 10  # 10 points along each edge of the simplex
        weights_list = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                k = n_grid - i - j
                w = np.array([i, j, k], dtype=float) / n_grid
                weights_list.append(w)

        # Also add the 3 vertices and edge midpoints for coverage
        for w in [np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([0, 0, 1.]),
                  np.array([0.5, 0.5, 0.]), np.array([0.5, 0., 0.5]),
                  np.array([0., 0.5, 0.5])]:
            if not any(np.allclose(w, ww, atol=0.01) for ww in weights_list):
                weights_list.append(w)

        # Phase 2: For each weight, do a local search from a random starting point
        rng = np.random.default_rng(0)
        max_evals_per_nm = 15

        for w in weights_list:
            if used >= self.budget or time.time() > deadline - 10:
                break

            # Start from a random point
            x0 = lo + rng.random(self.dim) * span
            nm_optimize(w, x0, maxiter=max_evals_per_nm)

        # Phase 3: Polish the archive with additional Nelder-Mead runs
        if archive_X and used < self.budget and time.time() < deadline - 10:
            # Select non-dominated points and refine them
            nondom_indices = []
            for i, f in enumerate(archive_F):
                if not is_dominated(f, archive_F):
                    nondom_indices.append(i)

            for idx in nondom_indices:
                if used >= self.budget or time.time() > deadline - 5:
                    break

                # Use a weight vector pointing toward this point's direction
                f = archive_F[idx]
                w = f / (np.sum(f) + 1e-10)
                nm_optimize(w, archive_X[idx], maxiter=10)

        # Phase 4: Final random sampling to fill gaps
        if used < self.budget and time.time() < deadline - 2:
            while used < self.budget and time.time() < deadline - 1:
                x = lo + rng.random(self.dim) * span
                f = problem(x)
                used += 1
                update_archive(x, f)

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
