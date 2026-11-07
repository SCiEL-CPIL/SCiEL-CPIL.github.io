"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, rng, x0, sigma0, max_evals, deadline):
        """Run CMA-ES from x0 with initial sigma sigma0 for up to max_evals evaluations."""
        n = self.dim
        calls = 0
        x = x0.copy()
        sigma = sigma0
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        lambda_ = 4 * n  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # State variables
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        while calls < max_evals and time.time() < deadline:
            # Generate population
            Y = np.array([x + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lambda_)])
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lambda_

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()

            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update mean
            x_new = weights @ Y[:mu]

            # Update step size (CSA)
            y_mean = (x_new - x) / (sigma * D) if sigma > 0 else np.zeros(n)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / D @ B.T @ y_mean
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - max_evals + 1))) < \
               (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update C matrix (rank-one and rank-mu update)
            pc = (1 - c_c) * pc + c_c * y_mean
            delta_sigma = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < \
                            (1.4 + 2 / (n + 1)) * chi_n else 0
            C = ((1 - c_1 - c_mu * delta_sigma) * np.eye(n)
                 + c_1 * (pc @ pc + (1 - delta_sigma) * (1 - c_c) * np.eye(n))
                 + c_mu * np.sum(weights[:i] * np.outer(Y[i] - x, Y[i] - x) / (sigma * D) ** 2
                                for i in range(mu)))
            # More efficient rank-mu update
            C = ((1 - c_1 - c_mu * delta_sigma) * np.eye(n)
                 + c_1 * (pc @ pc + (1 - delta_sigma) * (1 - c_c) * np.eye(n)))
            for i in range(mu):
                y_i = (Y[i] - x) / (sigma * D)
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            B = B.T

            x = x_new
            sigma = max(sigma, 1e-10)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Global exploration with large sigma
        # Try several random starting points with CMA-ES
        n_restarts = min(5, budget // 100)
        evals_per_restart = max(50, budget // (n_restarts + 2))

        for r in range(n_restarts):
            if total_calls >= budget or time.time() >= deadline:
                break

            x0 = rng.uniform(-5, 5, n)
            sigma0 = 2.0 if r == 0 else 1.5

            remaining = min(evals_per_restart, budget - total_calls)
            if remaining < 20:
                break

            bx, bf, calls_used = self._cma_es(func, rng, x0, sigma0, remaining, deadline)
            total_calls += calls_used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 2: Local refinement around the best point
        if total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining >= 20:
                x0 = best_x.copy()
                sigma0 = 0.5
                bx, bf, calls_used = self._cma_es(func, rng, x0, sigma0, remaining, deadline)
                total_calls += calls_used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        # Phase 3: Final polish with very small sigma
        if total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining >= 10:
                x0 = best_x.copy()
                sigma0 = 0.1
                bx, bf, calls_used = self._cma_es(func, rng, x0, sigma0, remaining, deadline)
                total_calls += calls_used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
