"""CMA-ES optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random warm-start for early AOCC credit."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # 270s safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # --- Phase 1: random sampling for early progress ---
        n_random = min(30, n)
        X = rng.uniform(-5, 5, size=(n_random, d))
        ys = np.array([func(x) for x in X])
        best_idx = int(np.argmin(ys))
        x_best = X[best_idx].copy()
        f_best = ys[best_idx]
        calls = n_random

        # --- Phase 2: CMA-ES ---
        # Population size
        lam = 10
        mu = lam // 2  # number of parents
        # Weights for rank-mu update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(0.4 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initialize CMA-ES state from best random point
        x_mean = x_best.copy()
        sigma = 2.0  # initial step size
        C = np.eye(d)
        C_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        # Cholesky of C (initially identity)
        C_chol = np.linalg.cholesky(C + 1e-12 * np.eye(d))

        remaining = n - calls
        for gen in range(remaining // lam + 1):
            if calls >= n:
                break
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if calls >= n:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_chol @ z)
                x_off = np.clip(y, -5, 5)
                f_off = func(x_off)
                calls += 1
                offspring.append((f_off, x_off, z))

            if len(offspring) < lam:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x_best = offspring[0][1].copy()

            # Select top-mu
            selected = offspring[:mu]
            x_w = np.sum([weights[i] * selected[i][1] for i in range(mu)], axis=0)
            z_w = np.sum([weights[i] * selected[i][2] for i in range(mu)], axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_chol.T @ z_w)
            norm_ps = np.linalg.norm(ps) / chi_d
            pc = (1 - c_c) * pc + c_c * (1.0 if norm_ps < (1.4 + 2 / (d + 1)) * (1 - c_c) else 0.0) * z_w

            # Update mean
            x_mean = x_w

            # Update covariance
            delta_diag = (norm_ps ** 2 - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_diag + c_mu) * C
            C -= c_1 * np.outer(pc, pc)
            C -= c_mu * np.sum([weights[i] * np.outer(selected[i][2], selected[i][2]) for i in range(mu)], axis=0)

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            try:
                C_chol = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-4 * np.eye(d)
                C = 0.5 * (C + C.T)
                C_chol = np.linalg.cholesky(C + 1e-12 * np.eye(d))

            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (norm_ps / chi_d - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
