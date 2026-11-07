"""Rewritten optimizer for BBOB f12 (Bent Cigar) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the bent cigar function on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (mu_eff - 1)) / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (2 + (mu_eff - 2) / (mu_eff + 1)) * (c_c + 1) / ((dim + 1.3) ** 2 + mu_eff))

        mean = np.zeros(dim)
        sigma = 2.0

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = mean.copy()
        best_f = float('inf')
        calls = 0

        rng = np.random.default_rng(0)

        def _eval(x):
            nonlocal calls
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls += 1
            return f_val

        # Initial evaluation
        f_val = _eval(mean)
        if f_val < best_f:
            best_f = f_val
            best_x = mean.copy()

        for gen in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            # Generate lambda normal vectors
            Z = rng.standard_normal((lam, dim))
            # Transform to correlated normals
            Y = mean[None, :] + sigma * (Z @ L.T)
            # Clip to box
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([_eval(y) for y in Y])

            # Track best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            mean_new = Y_sorted[:mu] @ weights[:, None]

            # Update sigma (CSA)
            step = mean_new - mean
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-16:
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ (step / (sigma * step_norm)))

                # Check h_sigma
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - 1)))
                if denom > 1e-16:
                    h_sigma = np.linalg.norm(p_sigma) / denom
                    if h_sigma < (1.4 + 2 / (dim + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance
            # Update p_c
            denom_c = np.sqrt(1 - (1 - c_c) ** (2 * (calls - 1)))
            if denom_c > 1e-16:
                h_c = np.linalg.norm(p_c) / denom_c < (1.4 + 2 / (dim + 1)) * chi_n
            else:
                h_c = True

            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (step / sigma)

            # Rank-one update
            v = (mean_new - mean) / sigma
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T - C)

            # Rank-mu update
            Y_diff = (Y_sorted[:mu] - mean) / sigma
            C = C + c_mu * (weights[:, None] * (Y_diff @ Y_diff.T))

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            mean = mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
