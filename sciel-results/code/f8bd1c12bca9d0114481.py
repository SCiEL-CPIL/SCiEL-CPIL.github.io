"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        n_calls_total = self.budget

        # Population size
        lam = 4 + int(3 * np.log(d))
        mu = lam // 2

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(d)
        sigma = 2.0
        B = np.eye(d)
        D = np.ones(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)

        # Parameters
        c_sigma = (mu + 2) / (d + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu / d) / (d + 4 + 2 * mu / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((d + 2) ** 2 + mu))

        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Track best
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Main loop
        while calls < n_calls_total:
            # Generate population
            pop = np.zeros((lam, d))
            fvals = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(d)
                y = B @ (D * z)
                candidate = x + sigma * y
                candidate = np.clip(candidate, -5, 5)
                pop[i] = candidate
                fvals[i] = func(candidate)
                calls += 1

            # Check if we've exhausted budget
            if calls >= n_calls_total:
                # Update best with last evaluations
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = pop[min_idx].copy()
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Compute weighted mean of selected parents
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += weights[i] * pop_sorted[i]

            # Update mean
            x_new = x + y_w
            x_new = np.clip(x_new, -5, 5)

            # Update step size
            if sigma > 1e-15:
                y_sigma = (y_w / sigma - p_c)
                chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
                h_sigma = min(1, np.linalg.norm(y_sigma) / np.sqrt(1 - np.exp(-2 * (calls - 1) / (n_calls_total + 1))) / chi_d)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
                sigma = np.clip(sigma, 1e-10, 100)

            # Update path for C
            if sigma > 1e-15:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu):
                diff = pop_sorted[i] - x
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Ensure C is symmetric and positive semi-definite
            C = 0.5 * (C + C.T)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                C = np.eye(d)
                D = np.ones(d)
                B = np.eye(d)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
