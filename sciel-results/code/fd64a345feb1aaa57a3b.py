"""Rewritten optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial population: sample around center and a few random points
        # f13 is a sharp ridge; center is a reasonable start
        x0 = np.zeros(dim)
        f0 = func(x0)

        # Try a few initial points to get a better starting center
        best_x = x0.copy()
        best_f = f0
        for i in range(min(10, budget)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2  # number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Step size
        sigma = 1.0
        # Initial mean
        mean = best_x.copy()
        # Covariance matrix
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chunk size for C update
        chunk_size = 20  # update C every 20 iterations

        calls_made = 11  # 1 initial + 10 random (approx)
        # Actually let's be more careful with call counting
        calls_made = 0

        # Reset and count properly
        # Let's just do a simpler approach: track calls

        # Actually, let me restructure to count calls properly
        # Let's use a wrapper

        call_count = [0]

        def count_func(x):
            call_count[0] += 1
            return func(x)

        # Redo initial evaluation with counting
        best_x = np.zeros(dim)
        best_f = count_func(best_x)
        calls_made = call_count[0]

        for i in range(min(10, budget - calls_made)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = count_func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        calls_made = call_count[0]

        # CMA-ES main loop
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        iter_count = 0
        last_c_update = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            population = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Compute C_sqrt and C_inv_sqrt periodically
                if iter_count % 20 == 0 and iter_count > 0:
                    # Update C_inv_sqrt and C_sqrt
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, 1e12)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                z = rng.standard_normal(dim)
                population[i] = np.clip(mean + sigma * C_sqrt @ z, -5, 5)

            # Evaluate population
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                fitness[i] = count_func(population[i])

            if calls_made >= budget or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]

            # Update mean
            new_mean = np.sum(w[:, None] * population[:mu], axis=0)
            new_mean = np.clip(new_mean, -5, 5)

            # Update evolution paths
            y = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            sigma_cond = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (iter_count + 1)) - 1 + 1e-16
            sigma *= np.exp(c_sigma / d_sigma * (sigma_cond - 1))

            # Update C
            if iter_count - last_c_update >= 20:
                y_c = (new_mean - mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                rank_mu = np.sum(w[:, None] * (population[:mu] - mean)[:, None] * (population[:mu] - mean)[None, :], axis=0) / sigma ** 2
                C = (1 + c_1 + c_mu - c_1 / dim) * C + c_1 * (np.outer(y_c, y_c) - np.eye(dim)) + c_mu * rank_mu
                # Symmetrize
                C = (C + C.T) / 2
                last_c_update = iter_count

            mean = new_mean
            iter_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
