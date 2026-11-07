"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES style optimizer with population-based search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a diverse starting point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        call_count = 1

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # population size
        c1 = (4 + np.log(dim)) / ((dim + 4) ** 2 + 4)
        cmu = min(1 - c1, 2 / ((dim + 1.3) ** 2 - 3))
        cs = (dim + 1.7) / (dim + 9.5)
        c_sigma = min(1.0, cs * (1 + cmu))
        d_sigma = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Precompute weights for recombination
        weights = np.exp(-np.arange(1, pop_size + 1) / (dim / 2))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        while call_count < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Update covariance matrix
            delta_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - call_count))) - 1) * cs * c_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-8)

            # Generate population
            population = []
            fvals = []
            for i in range(pop_size):
                if call_count + i >= budget:
                    break
                z = rng.standard_normal(dim)
                # Apply rotation via Cholesky of C
                L = np.linalg.cholesky(C)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                population.append(y)
                fvals.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(population) < 2:
                break

            fvals = np.array(fvals)
            population = np.array(population)

            # Sort by fitness
            order = np.argsort(fvals)
            population_sorted = population[order]
            fvals_sorted = fvals[order]
            weights_sorted = weights[:len(population_sorted)]

            # Update mean
            m_new = np.sum(weights_sorted[:, None] * population_sorted, axis=0)
            delta = (m_new - x) / sigma
            pc = (1 - c1) * pc + c1 * delta
            x = m_new

            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            diff = population_sorted - x
            C += cmu / mueff * np.sum(weights_sorted[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)

            # Update step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * delta

        return best_x, best_f
# EVOLVE-BLOCK-END
