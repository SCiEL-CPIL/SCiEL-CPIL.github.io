"""Optimized CMA-ES for BBOB f23 (Katsuura) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and covariance matrix."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Initialize
        x = np.zeros(dim)
        sigma = 3.0  # initial step size to cover the domain
        C = np.eye(dim)
        Csqrt = np.eye(dim)

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of parents
        lambda_pop = max(2 * mu + 1, dim + 1)  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Track best
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        # Initial evaluation of a few random points to get a better starting point
        rng = np.random.default_rng(42)
        for _ in range(min(10, budget - calls)):
            if time.time() > deadline:
                break
            x_init = rng.uniform(-5, 5, dim)
            f_init = func(x_init)
            calls += 1
            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()
                x = x_init.copy()

        # CMA-ES main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_pop):
                if time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                x_new = x + sigma * (Csqrt @ y)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)

            if len(xs) < 2:
                break

            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Mean update
            x_old = x.copy()
            x = np.sum(weights[:, None] * xs[:mu], axis=0)

            # Update covariance matrix
            # Compute step size
            y_mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (Csqrt.T @ y_mean) / chi_n
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lambda_pop)))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) +
                 c_mu * np.sum(weights[:, None, None] * np.outer(ys[:mu], ys[:mu]) * (Csqrt.T @ Csqrt), axis=0))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update Csqrt via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            Csqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(C - np.eye(dim)) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
