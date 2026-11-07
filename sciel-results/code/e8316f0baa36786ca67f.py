# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with multiple restarts for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Initial random sampling to find good starting points
        n_init = min(30, budget // 15)
        init_points = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            init_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        init_points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES refinement on top candidates
        # CMA-ES parameters
        def cma_es(x0, f0, budget_cma, dim, rng, deadline, best_x, best_f):
            """Run CMA-ES from starting point x0 with initial fitness f0."""
            n = dim
            # Population size
            lambda_ = max(4, int(4 + 3 * math.log(n)))
            # Initial sigma
            sigma = 0.5
            # CMA-ES weights
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            # Step size adaptation parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # Covariance matrix adaptation parameters
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # State variables
            x_mean = x0.copy()
            f_mean = f0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = math.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

            # Track best
            local_best_x = x0.copy()
            local_best_f = f0

            calls_used = 0
            for gen in range(budget_cma // lambda_ + 1):
                if time.time() > deadline:
                    break
                if calls_used >= budget_cma:
                    break

                # Generate population
                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    if calls_used >= budget_cma:
                        break

                    # Sample from N(x_mean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1

                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()

                    # Store for this generation
                    if not hasattr(cma_es, '_pop'):
                        cma_es._pop = []
                    cma_es._pop.append((fy, y))

                # Sort by fitness
                cma_es._pop.sort(key=lambda t: t[0])

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * cma_es._pop[i][1]
                x_mean = x_mean + (y_w - x_mean)

                # Update sigma
                y_sigma = (C_sqrt @ y_w) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update covariance
                y_c = (C_sqrt @ y_w)
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                delta_heav = 1 if np.linalg.norm(p_c) / math.sqrt(1 - (1 - c_c) ** (calls_used + 1)) / chi_n < 1.4 + c_c else 0
                C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + \
                    c_1 * (np.outer(p_c, p_c) + (1 - delta_heav) * C) + \
                    c_mu / sigma ** 2 * sum(weights[i] * np.outer(cma_es._pop[i][1] - x_mean, cma_es._pop[i][1] - x_mean) for i in range(mu))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    # Fallback: reinitialize
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_inv_sqrt = np.eye(n)

                # Update global best
                if local_best_f < best_f[0]:
                    best_f[0] = local_best_f
                    best_x[0] = local_best_x.copy()

                del cma_es._pop

            return local_best_x, local_best_f, calls_used

        # Run CMA-ES on top candidates
        top_k = min(5, len(init_points))
        calls_used_total = n_init

        for k in range(top_k):
            if time.time() > deadline:
                break
            if calls_used_total >= budget:
                break

            fx_init, x_init = init_points[k]

            # Allocate budget for this CMA-ES run
            remaining = budget - calls_used_total
            if k == top_k - 1:
                budget_cma = remaining
            else:
                budget_cma = max(20, remaining // (top_k - k))

            # Use mutable containers for best tracking
            best_x_container = [best_x.copy() if best_x is not None else x_init.copy()]
            best_f_container = [best_f]

            x_res, f_res, calls_cma = cma_es(x_init, fx_init, budget_cma, dim, rng, deadline, best_x_container, best_f_container)
            calls_used_total += calls_cma

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 3: If we still have budget, do a final local refinement
        remaining = budget - calls_used_total
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            for step in range(min(remaining, 50)):
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                sigma = np.clip(sigma, 1e-6, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
