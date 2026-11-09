"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, adaptive step size, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Time guard
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit

        # Initialize population around origin
        sigma = 2.0
        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Population size
        lambda_pop = 10
        mu = lambda_pop // 2  # number of selected parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evaluate initial population
        calls = 0
        best_x = mean.copy()
        best_f = float('inf')

        # Generate initial population
        pop = []
        fvals = []
        for i in range(lambda_pop):
            z = rng.standard_normal(dim)
            y = np.clip(mean + sigma * z, -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            pop.append(y)
            fvals.append(fy)

        # Sort by fitness
        order = np.argsort(fvals)
        pop = [pop[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Save old mean before updating
            mean_old = mean.copy()

            # Update mean
            y_w = np.sum([w[i] * pop[i] for i in range(mu)], axis=0)
            mean = y_w

            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (mean - mean_old)

            # Update C
            delta_heav = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / budget)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum([w[i] * np.outer(pop[i] - mean_old, pop[i] - mean_old) for i in range(mu)], axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2.0

            # Update pc
            pc = (1 - c_c) * pc + c_c * (mean - mean_old) / sigma

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 100)

            # Generate new population
            new_pop = []
            new_fvals = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for sampling from C
                try:
                    L = np.linalg.cholesky(C)
                    y = np.clip(mean + sigma * (L @ z), -5, 5)
                except np.linalg.LinAlgError:
                    # Fallback if Cholesky fails
                    y = np.clip(mean + sigma * z, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                new_pop.append(y)
                new_fvals.append(fy)

            if calls >= budget:
                break

            # Sort
            order = np.argsort(new_fvals)
            pop = [new_pop[i] for i in order]
            fvals = [new_fvals[i] for i in order]

        return best_x, best_f
# EVOLVE-BLOCK-END
