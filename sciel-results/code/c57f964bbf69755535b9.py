"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Multi-start: 10 independent CMA-ES runs
        n_starts = 10
        # Initial points spread across the box
        starts = []
        for i in range(n_starts):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 4.0)
            elif i == 2:
                x0 = np.full(dim, -4.0)
            else:
                x0 = rng.uniform(-4, 4, dim)
            starts.append(x0)

        # Track best overall
        best_x = None
        best_f = np.inf
        calls_made = 0

        # For each start, run a simplified CMA-ES
        # We'll do a (lambda, mu) CMA-ES with lambda=4, mu=2 for efficiency
        lambda_ = 4
        mu = lambda_ // 2

        # Pre-allocate weights for CMA-ES
        w = np.zeros(lambda_)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for s_idx, x0 in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Initialize CMA-ES for this start
            x_mean = x0.copy()
            sigma = 2.0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls_this_start = 0
            max_calls_per_start = max(50, budget // n_starts)

            while calls_this_start < max_calls_per_start and calls_made < budget:
                if time.time() > deadline - 5:
                    break

                # Generate lambda samples
                samples = np.zeros((lambda_, dim))
                fvals = np.zeros(lambda_)
                for j in range(lambda_):
                    z = rng.standard_normal(dim)
                    samples[j] = x_mean + sigma * (B @ (D * z))
                    samples[j] = np.clip(samples[j], -5, 5)
                    fvals[j] = func(samples[j])
                    calls_made += 1
                    calls_this_start += 1
                    if fvals[j] < best_f:
                        best_f = fvals[j]
                        best_x = samples[j].copy()

                # Sort by fitness
                order = np.argsort(fvals)
                y_sorted = samples[order]
                f_sorted = fvals[order]

                # Update mean
                x_new_mean = w @ y_sorted[:mu]

                # Update step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * ((x_new_mean - x_mean) / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_this_start + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

                # Update covariance matrix
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = ((1 + c_1 * delta_hsig - c_1 - c_mu) * C +
                     c_1 / c_c * (pc @ pc.T - np.eye(dim)) +
                     c_mu / (w @ w) * np.sum(w[:, None, None] * (y_sorted[:mu] - x_mean)[:, None] @ (y_sorted[:mu] - x_mean)[None, :] / sigma ** 2, axis=0))

                # Eigen decomposition for B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                x_mean = x_new_mean

                # Early termination check
                if f_sorted[0] - f_sorted[-1] < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
