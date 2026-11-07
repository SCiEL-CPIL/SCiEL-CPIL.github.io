"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size = 4*(1+3*5)=64
        mu = lambda_ // 2                # number of recombination weights = 32

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # CMa parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x = np.zeros(dim)  # start at origin
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        calls = 0
        best_x = x.copy()
        best_f = np.inf

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Generate random direction
                z = np.random.standard_normal(dim)
                # Apply transformation
                y = B @ (D * z)
                pop[i] = np.clip(x + sigma * y, -5, 5)

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            indices = np.argsort(fvals)
            pop_sorted = pop[indices]
            fvals_sorted = fvals[indices]

            # Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_sorted[i]

            # Update evolution paths
            y_diff = (x_new - x) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
            cs_norm = np.linalg.norm(pc) / chi_n
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff

            # Update step size (CSA)
            hsig = (cs_norm < (1.4 + 1 / dim) * chi_n)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            if hsig:
                delta_hs = (1 - c_c) / c_c
            else:
                delta_hs = 1
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (pop_sorted[i] - x) / sigma
                rankmu += weights[i] * np.outer(zi, zi)
            C = (1 + c_c * (delta_hs - 1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

            # Update eigen decomposition (lazy, every 10 generations)
            # For simplicity, we'll do it every generation in 5D
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-20))
            B = B2
            D = D2

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
