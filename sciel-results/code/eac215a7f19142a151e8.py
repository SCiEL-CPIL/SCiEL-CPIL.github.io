import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES with adaptive restarts for BBOB f13 (Sharp Ridge).
    CMA-ES is a state-of-the-art derivative-free optimizer that handles
    non-separable functions and ridges well through covariance adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        def cma_es(x0, f0, max_iter=200, sigma_init=1.0):
            """Run CMA-ES from starting point x0 with initial sigma."""
            nonlocal calls, best_f, best_x

            n = dim
            # Population size
            lambda_pop = int(4 + 3 * np.log(n))

            # CMA-ES parameters
            mu = lambda_pop // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size adaptation
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # Covariance adaptation
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initialize
            mean = x0.copy()
            sigma = sigma_init
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)

            # Precompute
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            hsig = (1 - np.prod(1 - c_sigma) ** (1 / (c_sigma * d_sigma))) ** 2 < 1 - 1e-16

            # Initialize population
            pop = []
            pop_f = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                x = mean + sigma * (C_sqrt @ z)
                f_val = safe_call(x)
                if f_val is None:
                    return
                pop.append(x)
                pop_f.append(f_val)

            # Track best
            idx_min = np.argmin(pop_f)
            if pop_f[idx_min] < best_f:
                best_f = pop_f[idx_min]
                best_x = pop[idx_min].copy()

            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort population
                order = np.argsort(pop_f)
                pop_sorted = [pop[i] for i in order]
                pop_f_sorted = [pop_f[i] for i in order]

                # Update mean
                y_mean = np.zeros(n)
                for i in range(mu):
                    y_mean += weights[i] * pop_sorted[i]
                y_mean /= sigma
                mean += c_c * (y_mean - mean)

                # Update covariance matrix
                y_c = np.zeros(n)
                for i in range(mu):
                    y_c += weights[i] * (pop_sorted[i] - mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

                # Hessian update
                delta_hsig = (1 - np.prod(1 - c_sigma) ** (1 / (c_sigma * d_sigma))) ** 2 < 1 - 1e-16
                hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (iteration + 1)) / chi_n) < (1.4 + 2 / (n + 1))

                C = (1 + c_1 * hsig * (2 - c_c - c_1) * np.outer(p_c, p_c) + c_mu * np.sum(weights) - c_mu) * C
                for i in range(mu):
                    z_i = (pop_sorted[i] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(z_i, z_i)

                # Ensure positive semi-definite
                C = 0.5 * (C + C.T)
                C = np.maximum(C, 0)
                np.fill_diagonal(C, np.maximum(np.diag(C), 1e-10))

                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_inv = np.linalg.inv(C)
                    C_inv_sqrt = np.linalg.cholesky(C_inv)
                except np.linalg.LinAlgError:
                    # Fallback: reset covariance
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    break

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.linalg.norm(y_mean) / chi_n - 1) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check convergence
                if np.linalg.norm(C - np.eye(n)) < 1e-10 and sigma < 1e-6:
                    break

                # Generate new population
                pop = []
                pop_f = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    f_val = safe_call(x)
                    if f_val is None:
                        return
                    pop.append(x)
                    pop_f.append(f_val)

                # Update global best
                idx_min = np.argmin(pop_f)
                if pop_f[idx_min] < best_f:
                    best_f = pop_f[idx_min]
                    best_x = pop[idx_min].copy()

        # Phase 1: Global exploration with random sampling
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: CMA-ES with restarts
        n_restarts = 3
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                x_start = best_x.copy()
                f_start = best_f
                sigma_init = 1.0
            else:
                # Perturb from best or new random start
                if rng.random() < 0.5:
                    x_start = best_x + rng.standard_normal(dim) * 0.5
                else:
                    x_start = rng.uniform(-5.0, 5.0, size=dim)
                f_start = safe_call(x_start)
                if f_start is None:
                    break
                sigma_init = 0.5

            cma_es(x_start, f_start, max_iter=150, sigma_init=sigma_init)

        return best_x, best_f
