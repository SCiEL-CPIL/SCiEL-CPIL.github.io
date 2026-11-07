"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance and step-size for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 1.0
        # Initial covariance: identity
        C = np.eye(n)
        # Eigen decomposition for efficient sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)

        # Population size
        lambda_pop = 4 + int(np.floor(3 * np.log(n)))
        # Learning rates
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hsig = 0.0

        calls_made = 1

        for generation in range(self.budget - 1):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Sample population
            ys = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                # Transform using eigen decomposition
                y = eigvecs @ (eigvals ** 0.5 * z)
                ys.append(y)
            Y = np.array(ys)

            # Evaluate population
            Fs = []
            for y in Y:
                if time.time() > deadline - 2.0:
                    # Fill remaining with best known
                    Fs.append(best_fx + 1e6)
                    continue
                x_try = np.clip(x + sigma * y, -5, 5)
                f_try = func(x_try)
                Fs.append(f_try)
                calls_made += 1
                if f_try < best_fx:
                    best_fx = f_try
                    best_x = x_try.copy()

            Fs = np.array(Fs)

            # Sort
            order = np.argsort(Fs)
            Y_sorted = Y[order]
            Fs_sorted = Fs[order]

            # Update mean
            mean_new = x + sigma * sum(weights[i] * Y_sorted[i] for i in range(mu))
            mean_new = np.clip(mean_new, -5, 5)

            # Update covariance
            y_w = Y_sorted[:mu]
            diff = y_w - Y_sorted[mu:]  # Approximation for efficiency
            # Proper rank-mu update
            C_new = (1 - c_1 - c_mu) * C
            for i in range(mu):
                C_new += weights[i] * np.outer(Y_sorted[i], Y_sorted[i])

            # Update step size
            ys_sigma = Y_sorted[:mu]
            p_sigma_new = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                          (eigvecs.T @ (eigvecs @ (sigma * mean_new - x))) / sigma
            # Simplified: use norm of weighted sum of steps
            p_sigma_new = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                          (sigma * (mean_new - x)) / sigma
            p_sigma_norm = np.linalg.norm(p_sigma_new)
            hsig = int(p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1))) / chi_n < (1.4 + 2 / (n + 1)) * chi_n)

            if hsig:
                p_c_new = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (sigma * (mean_new - x))
                C_new += (c_1 + c_mu) * C + (c_c * (2 - c_c) * mu_eff - c_1 - c_mu) * np.outer(p_c_new, p_c_new)
            else:
                p_c_new = (1 - c_c) * p_c

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update eigen decomposition periodically
            C = C_new
            if generation % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)

            p_sigma = p_sigma_new
            p_c = p_c_new
            x = mean_new
            fx = best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
