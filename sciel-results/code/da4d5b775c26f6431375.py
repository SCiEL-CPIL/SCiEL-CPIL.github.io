import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polish for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        deadline = time.time() + 320

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), -5, 5)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, maxfev):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            remaining = budget - calls
            if remaining < 5:
                return
            mf = min(maxfev, remaining)
            try:
                minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfev': mf, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                )
            except Exception:
                pass

        def cmaes(x0, sigma0, max_evals):
            """CMA-ES implementation."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            remaining = budget - calls
            if remaining < 10:
                return

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            eps = 1e-32

            evals_used = 0
            while evals_used < max_evals and calls < budget and time.time() < deadline:
                if time.time() > deadline:
                    break
                # Generate population
                B = np.linalg.cholesky(C)
                z = rng.standard_normal((lam, n))
                y = (z @ B.T).T  # lam x n
                x_pop = np.array([x_mean + sigma * y[i] for i in range(lam)])
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.array([wrapped_func(x_pop[i]) for i in range(lam)])
                evals_used += lam
                if np.any(np.isinf(f_pop)):
                    f_pop[np.isinf(f_pop)] = 1e10

                # Sort
                idx = np.argsort(f_pop)
                x_sorted = x_pop[idx]
                f_sorted = f_pop[idx]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()

                # Weighted mean
                x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

                # Success history
                y_mean = (x_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (n + 1)) * chi_n
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

                # Covariance adaptation
                delta_hsig = 1.0 if hsig else 0.0
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                    + c_1 * (np.outer(pc, pc) - delta_hsig * C) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - x_mean, x_sorted[:mu] - x_mean) / sigma ** 2, axis=0)

                # Sigma adaptation
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                x_mean = x_new

                # Convergence check
                if sigma < 1e-12 or np.all(np.abs(np.diag(C)) < 1e-24):
                    break

            return

        # === Phase 1: Initial sampling ===
        n_init = min(20, budget // 15)
        init_points = [np.zeros(dim)]
        # Structured points
        for d in range(dim):
            for v in [-3.0, 0.0, 3.0]:
                p = np.zeros(dim)
                p[d] = v
                init_points.append(p)
        # Random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-5, 5, dim))

        for p in init_points[:n_init]:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(p)

        if best_x is None:
            return np.zeros(dim), np.inf

        # === Phase 2: CMA-ES from best ===
        remaining = budget - calls
        cma_budget = int(remaining * 0.6)
        if cma_budget > 10:
            cmaes(best_x.copy(), 2.0, cma_budget)

        # === Phase 3: Polish ===
        if calls < budget and time.time() < deadline:
            polish(best_x, maxfev=max(10, (budget - calls) // 3))

        # === Phase 4: Second CMA-ES restart if budget remains ===
        if calls < budget - 20 and time.time() < deadline:
            # Restart from a different region
            x_restart = best_x + rng.normal(0, 3.0, dim)
            x_restart = np.clip(x_restart, -5, 5)
            remaining = budget - calls
            if remaining > 20:
                cmaes(x_restart, 1.5, int(remaining * 0.5))

        # === Phase 5: Final polish ===
        if calls < budget and time.time() < deadline:
            polish(best_x, maxfev=budget - calls)

        return best_x, best_f
