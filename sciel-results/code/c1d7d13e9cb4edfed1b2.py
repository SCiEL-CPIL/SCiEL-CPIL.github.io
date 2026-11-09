"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        c_sigma = (mu + 1) / ((dim + 2) ** 2 + mu)
        c_c = (4 + mu / c_sigma) / (dim + 4 + 2 * mu / c_sigma)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        def cma_es(x0, fx0, n_calls_remaining, rng):
            """Run CMA-ES starting from x0 with fx0."""
            if n_calls_remaining <= 0:
                return x0, fx0, 0

            x_mean = x0.copy()
            f_mean = fx0
            sigma = 1.0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            calls_used = 1  # fx0 already counted

            # Generate initial population
            def generate_population():
                nonlocal calls_used
                zs = rng.standard_normal((lam, dim))
                ps = x_mean + sigma * (zs @ (B * D))
                fs = np.array([func(np.clip(p, -5, 5)) for p in ps])
                calls_used += lam
                return ps, fs

            # Sort by fitness
            def sort_population(ps, fs):
                idx = np.argsort(fs)
                return ps[idx], fs[idx]

            # Main loop
            while calls_used < n_calls_remaining and (time.time() - start_time) < deadline - 5:
                # Check if we have enough budget for a full generation
                if calls_used + lam > n_calls_remaining:
                    break

                ps, fs = generate_population()
                ps, fs = sort_population(ps, fs)

                # Update best
                if fs[0] < f_mean:
                    f_mean = fs[0]
                    x_best = ps[0].copy()
                else:
                    x_best = x_mean.copy()

                # Update mean
                x_new_mean = sum(weights[i] * ps[i] for i in range(mu))

                # Update evolution paths
                diff = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_used + 1))
                           < (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
                delta_c = (c_c - c_c) if not h_sigma else c_c * (1 - c_c)

                # Update covariance matrix
                outer = np.zeros((dim, dim))
                for i in range(mu):
                    diff_i = (ps[i] - x_mean) / sigma
                    outer += weights[i] * np.outer(diff_i, diff_i)
                C = (1 - c_1 - c_mu) * C + c_1 * ((1 + delta_c) * np.outer(p_c, p_c) - np.eye(dim)) \
                    + c_mu * outer

                # Ensure C is positive semi-definite
                C = 0.5 * (C + C.T)
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                B, D = np.linalg.cholesky(C).T, None
                # Recompute B and D properly
                try:
                    L = np.linalg.cholesky(C)
                    B = L.T
                    D = 1.0 / np.sqrt(np.diag(C))
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
                    D = np.ones(dim)

                x_mean = x_new_mean
                sigma *= np.exp(c_sigma / lam * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
                sigma = np.clip(sigma, 1e-8, 100)

            return x_best, f_mean, calls_used

        # Multi-start strategy
        best_x = None
        best_f = np.inf
        total_calls = 0

        # First, do a quick random search to find a good starting point
        n_random = min(50, budget // 5)
        for i in range(n_random):
            if total_calls >= budget or (time.time() - start_time) > deadline - 10:
                break
            x_try = np.random.default_rng(42 + i).uniform(-5, 5, dim)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # Run CMA-ES from the best point
        remaining = budget - total_calls
        if remaining > 10:
            rng = np.random.default_rng(123)
            x_best, f_best, calls_used = cma_es(best_x, best_f, remaining, rng)
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # If we have budget left, try a few more restarts from different points
        if total_calls < budget - 20:
            for restart in range(3):
                if total_calls >= budget - 10 or (time.time() - start_time) > deadline - 10:
                    break
                rng = np.random.default_rng(1000 + restart)
                x_start = rng.uniform(-5, 5, dim)
                f_start = func(x_start)
                total_calls += 1
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()

                remaining = budget - total_calls
                if remaining > 20:
                    x_best, f_best, calls_used = cma_es(best_x, best_f, remaining, rng)
                    total_calls += calls_used
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best

        # Final polish with local search
        if total_calls < budget - 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while total_calls < budget and (time.time() - start_time) < deadline - 5:
                y = np.clip(x + sigma * np.random.default_rng(int(time.time() * 1000)).standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    best_x = x.copy()
                    best_f = fx
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
