import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Population-Based CMA-ES with multi-start for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(2024)

        calls_made = 0
        best_x = None
        best_f = np.inf

        def time_left():
            return deadline - time.time()

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return x, f

        # Phase 1: Initial random sampling to find a good starting point
        n_init = 20
        X_init = rng.uniform(-5, 5, size=(n_init, dim))
        Y_init = np.array([safe_call(x)[1] for x in X_init])
        init_best_idx = np.argmin(Y_init)
        best_f = Y_init[init_best_idx]
        best_x = X_init[init_best_idx].copy()

        # Phase 2: CMA-ES with multi-restart strategy
        # We'll run multiple CMA-ES instances, each starting from different points
        # but sharing the best result found so far

        def cma_es(x0, budget_cma, sigma0=None):
            """Run CMA-ES starting from x0 with given budget."""
            nonlocal calls_made, best_f, best_x

            if budget_cma <= 0 or time_left() < 5:
                return best_x, best_f

            # CMA-ES parameters
            N = max(4 + int(2 * np.log(dim)), 4)  # Population size
            if N > 12:
                N = 12
            lam = N
            mu = lam // 2  # Number of weights

            # Step size
            if sigma0 is None:
                sigma0 = 1.0
            sigma = sigma0

            # Initial mean
            mean = x0.copy()

            # Covariance matrix (identity initially)
            C = np.eye(dim)

            # Inverse square root of C (for sampling)
            C_inv_sqrt = np.eye(dim)

            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Learning rates
            c_sigma = (mu + 0.5) / (dim + 2.5 + mu)
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Track best
            local_best_f = best_f
            local_best_x = best_x.copy()

            # Initial population
            population = []
            fitness = []

            for gen in range(200):  # Max generations
                if calls_made >= budget or time_left() < 5:
                    break

                # Sample new population
                for i in range(lam):
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    x_new = mean + sigma * (C_inv_sqrt @ z)
                    x_new = np.clip(x_new, -5, 5)

                    f_new = func(x_new)
                    calls_made += 1

                    population.append(x_new)
                    fitness.append(f_new)

                    if f_new < local_best_f:
                        local_best_f = f_new
                        local_best_x = x_new.copy()

                # Sort by fitness
                indices = np.argsort(fitness)
                population = [population[i] for i in indices]
                fitness = [fitness[i] for i in indices]

                # Update mean
                mean_old = mean.copy()
                mean = sum(weights[i] * population[i] for i in range(mu))

                # Update evolution paths
                # p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    C_inv_sqrt @ ((mean - mean_old) / sigma)
                )

                # Check if sigma should be adapted
                chi_n = dim * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1. / dim) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # p_c
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 1. / dim) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                    C_inv_sqrt @ ((mean - mean_old) / sigma)
                )

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(population[i] - mean_old, population[i] - mean_old) for i in range(mu))
                C = (1 + c_1 * (mu_eff - 2 + delta_h) - c_1) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is positive definite and compute inverse square root
                # Regularize if needed
                C += 1e-12 * np.eye(dim)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

            return best_x, best_f

        # Multi-start CMA-ES
        # First, do a short CMA-ES from the best initial point
        budget_per_run = max(50, (budget - calls_made) // 3)
        cma_es(best_x, budget_per_run, sigma0=1.5)

        # Then try from a few other good initial points
        sorted_init = np.argsort(Y_init)
        for idx in sorted_init[1:4]:
            if calls_made >= budget or time_left() < 10:
                break
            remaining = budget - calls_made
            if remaining < 30:
                break
            # Start from a point near the current best but with some perturbation
            x_start = best_x + rng.normal(0, 0.5, size=dim)
            x_start = np.clip(x_start, -5, 5)
            cma_es(x_start, remaining, sigma0=0.8)

        # Final refinement: small CMA-ES with small sigma
        if calls_made < budget and time_left() > 10:
            remaining = budget - calls_made
            if remaining >= 20:
                cma_es(best_x, remaining, sigma0=0.1)

        return best_f
