"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return calls_used < budget and time.time() < deadline - 5.0

        def check_time_early():
            return calls_used < budget and time.time() < deadline - 20.0

        rng = np.random.default_rng(12345)

        # CMA-ES implementation
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = max(4, int(4 + 3 * np.log(dim)))
        # mu = floor(lam / 2)
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        # Step size
        sigma = 2.0  # initial step size for [-5,5]^5
        # Mean
        mean = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        C_sqrt_inv = np.eye(dim)
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Cache for matrix operations
        d_corr = 1 + c_c * (2 - c_c)

        def sample_from_C(sigma_val):
            """Sample from N(0, C) using Cholesky decomposition."""
            # Recompute Cholesky if needed
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            z = rng.standard_normal(dim)
            return L @ z

        def update_C_sqrt_inv():
            """Update inverse square root of C."""
            nonlocal C_sqrt_inv
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Initial population evaluation
        pop = []
        pop_f = []
        
        for i in range(lam):
            if not check_time_early():
                break
            z = rng.standard_normal(dim)
            x = mean + sigma * z
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            pop.append(x)
            pop_f.append(fx)
            try_update(x, fx)

        # Main CMA-ES loop
        while check_time() and len(pop) >= lam:
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in indices]
            pop_f_sorted = [pop_f[i] for i in indices]

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (pop_sorted[i] - mean) / sigma
            mean_old = mean.copy()
            mean = mean + sigma * y_w
            mean = np.clip(mean, -5, 5)

            # Update sigma
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (pop_sorted[i] - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.chi2(dim)) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C
            y_c = np.zeros(dim)
            for i in range(mu):
                y_c += weights[i] * (pop_sorted[i] - mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * np.eye(dim))
            for i in range(mu):
                y_i = (pop_sorted[i] - mean_old) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            C = (C + C.T) / 2  # Symmetrize

            # Update inverse square root
            update_C_sqrt_inv()

            # Generate new population
            new_pop = []
            new_pop_f = []
            for i in range(lam):
                if not check_time():
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt_inv @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                new_pop.append(x)
                new_pop_f.append(fx)
                try_update(x, fx)

            if len(new_pop) < lam:
                # Pad with best existing points if we ran out of budget
                while len(new_pop) < lam:
                    new_pop.append(pop_sorted[0].copy())
                    new_pop_f.append(pop_f_sorted[0])

            pop = new_pop
            pop_f = new_pop_f

        # Phase 2: Local refinement around best point
        if check_time() and not np.isnan(best_x).any():
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.3

            no_improve = 0
            max_no_improve = 50

            while check_time() and no_improve < max_no_improve:
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma_local *= 1.05
                    no_improve = 0
                else:
                    sigma_local *= 0.9
                    no_improve += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
