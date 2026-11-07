"""CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, covariance adaptation, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 10
        mu = lam // 2  # 5 parents

        # Initial parameters
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        # Precompute C^{-1/2} for generating samples
        C_inv_sqrt = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(dim)))

        # CMA-ES parameters
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, (mu - 2) / (dim + mu)) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        hs = 0.0
        hs_init = (1 + c_c) ** 0.5  # threshold for sigma adaptation

        # Track best
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1

        # Precompute weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        for gen in range(1, 10000):
            # Check deadline
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                ys[i] = np.clip(y, -5, 5)

            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    fs[i] = best_f  # fill with best
                    continue
                fs[i] = func(ys[i])
                calls_made += 1

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update sigma path
            diff = x - x_old
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * diff

            # Hessian norm
            p_sigma_norm = np.linalg.norm(p_sigma)

            # Update hs
            hs = (1 - c_sigma) * hs + (2 * c_sigma / (1 - c_sigma) - 1) * (
                p_sigma_norm / chi_n > (1.4 + 2 / (dim + 1))
            )

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update C path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * diff

            # Update covariance
            delta_hs = (1 - hs) * (2 * c_c / (1 - c_c) - 1)
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) - c_1 + delta_hs * c_c) * C
            C += c_1 * (p_c[:, None] * p_c[None, :] - C)
            C += c_mu / sigma ** 2 * np.sum(
                weights[:, None, None] * (ys_sorted[:mu] - x_old)[:, None, :] *
                (ys_sorted[:mu] - x_old)[None, :, :], axis=0
            )

            # Ensure symmetry
            C = (C + C.T) / 2

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Recompute C^{-1/2}
            try:
                C_inv_sqrt = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(dim)))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Clip mean to box
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
