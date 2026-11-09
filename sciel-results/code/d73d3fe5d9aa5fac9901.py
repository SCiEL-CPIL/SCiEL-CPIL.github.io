"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Population-based stochastic search for BBOB-style convex functions.

    Strategy:
      1. Cheap LHS spread (+ origin) to establish a good early best
         (AOCC rewards early progress).
      2. CMA-ES population search from the best sample (rotation-adaptive,
         handles the hidden rotation/shift well).
      3. Final L-BFGS-B polish on the convex Sphere function.
    An explicit wall-clock deadline guard keeps us safely under the limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(0)

        # Wrap func to count calls and clip to the box.
        call_count = [0]

        def f(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        def time_ok():
            return time.time() < deadline and call_count[0] < budget

        # ---- Phase 1: cheap spread to get a good early best ----
        # Latin Hypercube + explicit origin (Sphere minima after shift are
        # often near center). 31 points total.
        n_lhs = 30
        x0 = rng.uniform(-5.0, 5.0, (n_lhs, dim))
        # origin
        samples = np.vstack([x0, np.zeros(dim)])
        # shuffle to interleave origin early in evaluation order
        perm = rng.permutation(len(samples))
        samples = samples[perm]

        best_x = None
        best_f = np.inf
        for s in samples:
            if not time_ok():
                break
            fs = f(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # ---- Phase 2: CMA-ES from the best sample ----
        # Use a moderate population; CMA-ES adapts to rotation/shift.
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # ~15 for dim=5
        sigma = 1.0
        # CMA-ES state
        mean = best_x.copy()
        C = np.eye(dim)
        Cinv = np.eye(dim)
        # Precompute Cholesky
        L = np.linalg.cholesky(C)
        # Step-size control state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + dim / 3) / (dim + 4 + lambda_)
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_1 = (2 / ((dim + 1.3) ** 2 + lambda_)) * (1 + 2 / max(dim - 1, mu_eff - 1))
        c_mu = (2 / ((dim + 2) ** 2 + lambda_)) * (
            2 * c_1 - 1 + 2 * min(w[0], 1.0 / mu_eff - 1)
        )
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while time_ok():
            # Generate population
            population = np.empty((lambda_, dim))
            for i in range(lambda_):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * L @ z
                population[i] = np.clip(y, -5.0, 5.0)
            # Evaluate
            values = np.empty(lambda_)
            for i in range(lambda_):
                if not time_ok():
                    values[i] = np.inf
                    break
                values[i] = f(population[i])
            # Track global best
            idx_best = int(np.argmin(values))
            if values[idx_best] < best_f:
                best_f = values[idx_best]
                best_x = population[idx_best].copy()

            if not time_ok():
                break

            # Sort by fitness
            order = np.argsort(values)
            y_sorted = population[order]
            # Weighted mean
            y_w = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            # Update CMA-ES state
            y_tilde = (y_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * y_tilde
            p_c = (1 - c_c) * p_c + np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * y_tilde
            h_sig = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count[0] // lambda_ + 1)))
                < (1 + 2 / (dim + 1)) * chi_n
            )
            C = (
                (1 - c_1 - c_mu + (not h_sig) * c_c * (2 - c_c) * c_1) * C
                + c_1 * (np.outer(p_c, p_c) - C)
                + c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu] - mean, y_sorted[:mu] - mean) / sigma ** 2, axis=0)
            )
            # Symmetrize and re-cholesky periodically
            C = 0.5 * (C + C.T)
            # Ensure positive definiteness
            C += 1e-9 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            mean = y_w
            # Update sigma (1/5-like via p_sigma)
            sigma *= np.exp(
                c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-6, 20.0)

        # ---- Phase 3: final L-BFGS-B polish on the convex function ----
        # Use remaining budget carefully.
        if time_ok() and best_x is not None and call_count[0] < budget:
            # Count remaining budget
            remaining = budget - call_count[0]
            # L-BFGS-B uses ~ (dim+1) evals per iteration; be conservative.
            max_iter = max(5, min(remaining // (dim + 1), 60))
            try:
                res = minimize(
                    f,
                    best_x,
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": max_iter, "gtol": 1e-12},
                )
                if res.success or res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
