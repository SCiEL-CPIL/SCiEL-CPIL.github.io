"""Optimizer for BBOB f21 (Gallagher 101 peaks) – fresh design."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with global scan initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Global scan to find a good starting point ---
        # Use a fraction of budget for initial exploration
        n_scan = min(50, budget // 10)
        x_best = np.full(dim, np.nan)
        f_best = np.inf

        # Start with center and random points
        candidates = [np.zeros(dim)]
        for _ in range(n_scan - 1):
            candidates.append(rng.uniform(-5, 5, dim))

        for x in candidates:
            if time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < f_best:
                f_best = fx
                x_best = x_clipped.copy()

        # --- Phase 2: CMA-ES optimization ---
        # Population size
        n = max(4, int(4 + 3 * np.log(dim)))
        # Initial sigma: large to cover multiple basins
        sigma = 2.0
        mean = x_best.copy()
        f_mean = f_best

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Step sizes
        cs = (n + 2) / (dim + n + 2)
        ds = 1 + 2 * max(0, np.sqrt((n + 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / n) / (dim + 4 + 2 * n / dim)
        c1 = (2 / ((dim + 1.3) ** 2 + n)) * (1 - cc)
        cmu = min(0.5 - c1, 2 / ((dim + 2) ** 2 + n))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = n_scan
        remaining = budget - calls_made

        for gen in range(remaining):
            if time.time() > deadline:
                break

            # Generate population
            # Update step size
            y_sigma = (p_c / np.sqrt(cs * (1 - cs)) - p_sigma) / (sigma * chi_n) if sigma > 0 else np.zeros(dim)
            h_sigma = np.linalg.norm(y_sigma) < (1.4 + 1 / dim) * np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            p_sigma = (1 - cs) * p_sigma + h_sigma * y_sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Generate population
            population = []
            fitnesses = []
            for i in range(n):
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for correlated samples
                # For simplicity and speed, we'll use a simplified covariance update
                x_new = mean + sigma * z
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                calls_made += 1
                population.append(x_new)
                fitnesses.append(fx)

            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fitnesses)
            population = [population[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Update mean with weighted recombination
            weights = np.log(n / 2 + 0.5) - np.log(np.arange(1, n + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights[0] ** 2 / (weights ** 2).sum())

            new_mean = np.zeros(dim)
            for i in range(n):
                new_mean += weights[i] * population[i]
            mean = new_mean
            f_mean = fitnesses[0]

            # Update covariance
            y = np.array(population[:int(mu_eff)]).T
            y = (y - mean[None, :].T) / sigma
            delta_h = (1 - h_sigma) * cc * (1 - cc)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim)) + cmu * (y @ y.T)

            # Update evolution paths
            y_sigma = (mean - new_mean) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * y_sigma
            mean = new_mean

        return x_best, f_best
# EVOLVE-BLOCK-END
