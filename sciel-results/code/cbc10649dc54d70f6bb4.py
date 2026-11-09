"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale CMA-ES variant with deterministic geometric decay."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES state
        C = np.eye(dim)
        # Initial sigma: large to explore the whole box
        sigma = 4.0

        # Learning rates (standard CMA-ES values)
        mu = max(1, dim // 2)  # number of best points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1
        stagnation_count = 0
        plateau_threshold = 1e-6

        # Geometric decay schedule for sigma
        # We'll shrink sigma by a factor each generation to ensure convergence
        # But we also adapt sigma via CMA-ES mechanism

        for gen in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda samples
            lam = max(4, 2 * dim)  # population size
            ys = rng.standard_normal((lam, dim))
            # Apply CMA-ES covariance matrix via Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            xs = x + sigma * (ys @ L.T)
            # Clip to box
            xs = np.clip(xs, -5, 5)

            # Evaluate
            fs = np.array([func(xi) for xi in xs])
            calls_made += lam

            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Weighted mean of best mu points
            y_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            x_mean = x + sigma * (y_mean @ L.T)
            x_mean = np.clip(x_mean, -5, 5)

            # Update sigma (CMA-ES adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ y_mean) / sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C (rank-one and rank-mu)
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1 + 2 / dim) * chi_n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (L @ y_mean)
            C = (1 + c_1 * h_sigma * (2 - h_sigma) + c_mu) * C \
                - c_1 * np.outer(p_c, p_c) \
                - c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]) * (2 - h_sigma), axis=0)

            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)

            # Move to weighted mean
            x = x_mean
            fx = np.sum(weights * fs_sorted[:mu])  # Approximate

            # Stagnation detection
            if best_f - fs_sorted[0] < plateau_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # If stagnating for too long, restart with smaller sigma
            if stagnation_count > 50:
                x = best_x.copy()
                sigma = max(sigma * 0.5, 1e-4)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_count = 0

            # Deterministic geometric decay: slowly shrink sigma to ensure convergence
            sigma *= 0.995

        return best_x, best_f
# EVOLVE-BLOCK-END
