"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant with adaptive population size for BBOB f13 (sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start with a reasonable population size, adaptively shrink
        lam = max(4, int(4 + 2 * np.log(dim)))  # initial population size
        mean = np.array([0.0] * dim)
        sigma = 2.0

        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Cholesky-like inverse sqrt for sampling: use eigen decomposition initially
        # For efficiency, we'll maintain C and use its square root via eigendecomposition
        # Actually, let's use the standard CMA-ES update with C and sigma

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Learning rates
        c_sigma = (1 + 2 / max(1, dim - 2) + lam / (2 * dim)) / (dim + 4)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) - 2) / ((dim + 2) ** 2 + lam))

        # mu for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / weights.sum() ** 2

        # Adjust learning rates for mu_eff
        c_1 = c_1 * (2 + 2 / mu_eff)
        c_mu = c_mu * (mu_eff / 2)

        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Precompute some constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Adaptive population size: shrink lam as we converge
        # We'll reduce lam by a factor when progress stalls or as budget depletes
        lam_current = lam
        min_lam = 4

        for gen in range(budget - 1):
            # Adaptively shrink population size
            # Reduce lam gradually; more aggressive reduction in later generations
            progress_ratio = gen / max(1, budget - 1)
            # Shrink population: start full, reduce to min_lam
            # Use a smooth schedule
            if progress_ratio < 0.3:
                lam_current = lam
            elif progress_ratio < 0.6:
                lam_current = max(min_lam, int(lam * (1 - (progress_ratio - 0.3) / 0.3 * 0.5)))
            else:
                lam_current = max(min_lam, int(lam * (0.5 - (progress_ratio - 0.6) / 0.4 * 0.35)))

            lam_current = max(min_lam, min(lam, lam_current))

            # Ensure lam_current is at least 2
            lam_current = max(2, lam_current)

            # Update mu based on current lam
            mu_current = max(1, lam_current // 2)
            weights_current = np.log(mu_current + 1) - np.log(np.arange(1, mu_current + 1))
            weights_current /= weights_current.sum()
            mu_eff_current = 1 / weights_current.sum() ** 2

            # Recompute learning rates for current lam
            c_sigma_cur = (1 + 2 / max(1, dim - 2) + lam_current / (2 * dim)) / (dim + 4)
            c_c_cur = (4 + lam_current / dim) / (dim + 4 + 2 * lam_current / dim)
            c_1_cur = 2 / ((dim + 1.3) ** 2 + lam_current) * (2 + 2 / mu_eff_current)
            c_mu_cur = min(1 - c_1_cur, 2 * ((lam_current / 2 + 1) - 2) / ((dim + 2) ** 2 + lam_current)) * (mu_eff_current / 2)

            # Sample offspring
            # Use eigen decomposition of C for sampling (simple but effective for small dim)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Clip negative eigenvalues due to numerical issues
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            B = eigenvectors
            D = np.sqrt(eigenvalues)

            # Generate normal random numbers
            z = np.random.randn(lam_current, dim)
            # Transform: mean + sigma * B @ diag(D) @ z
            # z @ (B * D) gives shape (lam, dim)
            y = mean[None, :] + sigma * (z @ (B * D[None, :]))

            # Clip to box
            y = np.clip(y, -5, 5)

            # Evaluate offspring
            f_y = np.array([func(y[i]) for i in range(lam_current)])
            calls_made += lam_current

            # Sort by fitness
            order = np.argsort(f_y)
            y_sorted = y[order]
            f_sorted = f_y[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Update evolution path for sigma
            y_w = (y_sorted[0] - mean) / sigma
            ps = (1 - c_sigma_cur) * ps + np.sqrt(c_sigma_cur * (2 - c_sigma_cur) * mu_eff_current) * y_w

            # Update mean
            mean_new = mean + sigma * (weights_current @ y_sorted[:mu_current, :])

            # Update covariance
            y_diff = y_sorted[:mu_current, :] - mean
            delta_mean = (mean_new - mean) / sigma

            C = (1 + c_1_cur + c_mu_cur - c_c_cur * c_c_cur * (1 - c_c_cur) / c_c_cur) * C \
                + c_1_cur * np.outer(delta_mean, delta_mean) \
                + c_mu_cur / sigma ** 2 * (y_diff.T @ (weights_current * y_diff))

            # Symmetrize C
            C = (C + C.T) / 2

            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma_cur / chi_n * (norm_ps / chi_n - 1))

            # Update pc
            pc = (1 - c_c_cur) * pc + c_c_cur * delta_mean

            # Regularize C: ensure positive definite
            # Add small identity if needed
            eigvals_check = np.linalg.eigvalsh(C)
            if eigvals_check.min() < 1e-10:
                C += 1e-10 * np.eye(dim)

            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)

            mean = mean_new

            # Early stopping: if sigma is very small and no progress
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
