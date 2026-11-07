"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Multi-method optimizer: coarse grid + differential evolution + local CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Coarse grid search to find a good starting region
        # Use a 3^5 = 243 point grid
        grid_points = []
        for i in range(3 ** dim):
            x = np.zeros(dim)
            rem = i
            for d in range(dim):
                x[d] = -5 + (2 * rem) / 2  # -5, 0, 5
                rem //= 3
            grid_points.append(x)
        grid_points = np.array(grid_points)

        # Shuffle to avoid bias
        perm = rng.permutation(len(grid_points))
        for idx in perm:
            if not time_ok() or calls_used >= budget:
                break
            eval_point(grid_points[idx])

        # Phase 2: Differential Evolution from best point
        if time_ok() and calls_used < budget:
            # Use scipy's differential evolution with a limited budget
            # Set maxiter and maxfev to stay within budget
            remaining = budget - calls_used
            if remaining > 100:
                try:
                    # Wrap func to track calls and enforce deadline
                    def wrapped_func(x):
                        if not time_ok() or calls_used >= budget:
                            return 1e10
                        return eval_point(x)

                    # DE with bounded constraints
                    bounds = [(-5, 5)] * dim
                    result = differential_evolution(
                        wrapped_func,
                        bounds,
                        maxiter=50,
                        maxfev=min(remaining - 10, 200),
                        popsize=10,
                        mutation=(0.3, 0.8),
                        recombination=0.8,
                        tol=1e-6,
                        seed=42,
                        polish=False,
                        init='sobol',
                        atol=1e-8,
                        update_interval=1,
                    )
                    if result.x is not None and result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Phase 3: Local CMA-ES refinement around best point
        if time_ok() and calls_used < budget and best_x is not None:
            x0 = best_x.copy()
            sigma = 0.5
            lambda_ = 4 + int(3 * np.log(dim))  # ~12 for dim=5
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - cc)
            cmu = (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (
                min(1, c1) - 1
            )
            cmu = max(0, min(cmu, 0.5))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

            x_mean = x0.copy()
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0

            max_generations = (budget - calls_used) // lambda_
            for gen in range(max_generations):
                if not time_ok() or calls_used >= budget:
                    break

                # Generate population
                y = rng.standard_normal((lambda_, dim))
                z = (y * D) @ B.T
                x_pop = x_mean + sigma * z
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.empty(lambda_)
                for i in range(lambda_):
                    if not time_ok() or calls_used >= budget:
                        f_pop[i] = best_f
                    else:
                        f_pop[i] = eval_point(x_pop[i])

                # Sort by fitness
                idx_sorted = np.argsort(f_pop)
                y_sorted = y[idx_sorted]
                f_sorted = f_pop[idx_sorted]

                # Weighted mean
                x_new = np.sum(weights[:, None] * x_pop[idx_sorted[:mu]], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D @ B.T @ (
                    (x_new - x_mean) / sigma
                )
                hsig = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1) / (dim + mueff)))
                    < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1)
                )
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (
                    (x_new - x_mean) / sigma
                )
                delta = (1 + hsig * cs * (1 - cs)) * (np.outer(pc, pc) - np.eye(dim))
                C_update = (
                    c1 * delta
                    + cmu * np.sum(
                        weights[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]),
                        axis=0,
                    )
                    - (c1 + cmu) * np.eye(dim)
                )
                eigvals, eigvecs = np.linalg.eigh(B @ (D ** 2) @ B.T + C_update)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                x_mean = x_new

                # If no improvement for a while, reduce sigma
                if f_sorted[0] >= best_f and sigma > 0.01:
                    sigma *= 0.95

                # Early stop if sigma is very small
                if sigma < 1e-6:
                    break

        # Phase 4: Final local search with small steps
        if time_ok() and calls_used < budget and best_x is not None:
            sigma_local = 0.1
            for _ in range(min(50, budget - calls_used)):
                if not time_ok() or calls_used >= budget:
                    break
                x_try = best_x + sigma_local * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = eval_point(x_try)
                if f_try < best_f:
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.7
                if sigma_local < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
