"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f1 (Sphere) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        c_1 = (2 + 2 * min(1, (lambda_ - 1) / (n + 2)) / (n + 2)) / ((n + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, (2 * (mu_eff - 0.5)) / ((n + 2) ** 2 + mu_eff))

        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        sigma = 2.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)  # We'll use a simple approach

        # Precompute C_sqrt using Cholesky-like update
        # For simplicity and speed, we'll use the standard CMA-ES with
        # covariance matrix updates but keep it numerically stable

        def update_cma(x, fx, y, fy, sigma, p_sigma, p_c, C, C_sqrt_inv, rng):
            nonlocal calls
            # Select top mu
            ys = sorted(zip(fy, y), key=lambda t: t[0])
            y_best = np.array([t[1] for t in ys[:mu]])
            f_best = np.array([t[0] for t in ys[:mu]])

            y_mean = np.sum(weights[:, None] * y_best, axis=0)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (y_mean - x) / sigma)

            # Update C
            y_mean_diff = y_mean - x
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean_diff / sigma

            delta = 1 + (1 - np.exp(-np.linalg.norm(p_c) / np.sqrt(n))) * (1 - np.exp(-2 * (np.linalg.norm(p_c) / np.sqrt(n) - n - 1)))

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - np.exp(-2 * np.linalg.norm(p_c) / np.sqrt(n))) * np.eye(n)) + c_mu * np.sum(weights[:, None, None] * np.outer(y_best - x, y_best - x), axis=0) / sigma ** 2

            # Regularize C
            C = (C + C.T) / 2
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_sqrt_inv
            eigvals_inv = 1.0 / np.sqrt(eigvals)
            C_sqrt_inv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Update x
            x_new = x + y_mean_diff

            return x_new, sigma, p_sigma, p_c, C, C_sqrt_inv

        # Simple CMA-ES loop
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate lambda samples
            ys = []
            fys = []
            for j in range(lambda_):
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * (C_sqrt_inv @ z), -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fys.append(fy)

            ys = np.array(ys)
            fys = np.array(fys)

            # Update
            x, sigma, p_sigma, p_c, C, C_sqrt_inv = update_cma(x, fx, ys, fys, sigma, p_sigma, p_c, C, C_sqrt_inv, rng)

            # Update best
            min_idx = np.argmin(fys)
            if fys[min_idx] < fx:
                fx = fys[min_idx]
                x_best = ys[min_idx]
            else:
                x_best = x

        return x, fx
# EVOLVE-BLOCK-END
