"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin: 300s limit, leave 15s

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        # Initial population
        best_x = None
        best_f = np.inf
        calls_made = 0

        # First, do a quick random search to find a good starting point
        rng = np.random.default_rng(42)
        for i in range(min(10, budget)):
            x_try = rng.uniform(-4, 4, dim)
            f_try = func(x_try)
            calls_made += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()

        # CMA-ES main loop
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)

            for i in range(lambda_):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                # Transform using C matrix
                # x + sigma * B @ (D * z)
                pop[i] = np.clip(x + sigma * (B @ (D * z)), -5, 5)
                pop_f[i] = func(pop[i])
                calls_made += 1

            if calls_made >= budget:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = pop[order]
            pop_f_sorted = pop_f[order]

            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of top mu
            y_w = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Step-size control
            y_sigma = (B @ (D * y_w)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - budget + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-12)

            # Position update
            x = x + sigma * y_w

            # Covariance matrix update
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_w
            C = (1 + c_1 * (hsig - 1) + c_mu) * C \
                - c_1 * np.outer(pc, pc) \
                - c_mu / (sigma ** 2) * np.sum(weights[:, None] * np.outer(pop_sorted[:mu] - x, pop_sorted[:mu] - x), axis=0)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            try:
                # Eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
            except np.linalg.LinAlgError:
                # Fallback: reset C
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

            # Check for convergence
            if sigma < 1e-10 or np.max(D) / np.min(D) > 1e14:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
