"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s out of 300s limit

        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES
        x = np.zeros(dim)  # start at origin (shifted by hidden offset)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = 4 * (int(np.log(dim)) + 1)  # population size
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 + 2.0 / (dim + 1)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9 + 2.0 / c1)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cc = (4 + dim / 4) / (dim + 4 + 2 / c1)

        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        step_size = np.sqrt(2 - c1) / (dim * np.sqrt(2 - c1))  # chi_N approximation
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = x + sigma * (B @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                fvals[i] = func(pop[i])
                calls_made += 1
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_fx:
                best_fx = fvals[min_idx]
                best_x = pop[min_idx].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y = pop[order]
            mu = lambda_ // 2 + 1  # number of recombination weights

            # Weights
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)

            # Mean update
            x_new = sum(w[i] * y[i] for i in range(mu))

            # Update evolution paths
            d = x_new - x
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) / sigma * d
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / (1 - (1 - cs) ** (budget - calls_made))) < \
                   (1.4 + 2 / (dim + 1)) * chi_n
            ps = (1 - ds) * ps + hsig * np.sqrt(ds * (2 - ds) * mueff) * d / sigma
            norm_ps = np.linalg.norm(ps)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * ds * (2 - ds)
            C = (1 - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + (delta_hsig - 1) * C) + \
                cmu / sigma ** 2 * sum(w[i] * np.outer(y[i] - x, y[i] - x) for i in range(mu))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / (dim + cc) * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x_new

            # Track best
            if fvals[min_idx] < best_fx:
                best_fx = fvals[min_idx]
                best_x = pop[min_idx].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
