"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like with coordinate restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Coarse random search to find a good starting point
        # Use 20% of budget for initial exploration
        phase1_budget = min(100, budget // 5)
        x = np.random.default_rng(0).uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        for _ in range(phase1_budget - 1):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # Phase 2: CMA-ES from best point
        x = best_x.copy()
        sigma = 1.0
        n = dim
        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Inverse square root via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals_inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
        C_inv_sqrt = (eigvecs * eigvals_inv_sqrt) @ eigvecs.T

        remaining = budget - calls
        best_fx_cma = best_fx
        best_x_cma = best_x.copy()

        for gen in range(max(1, remaining // lambda_)):
            if not time_ok():
                break

            # Generate lambda samples
            Y = rng.standard_normal((lambda_, n))
            X = x[:, None] + sigma * (Y @ C_inv_sqrt.T)  # Broadcasting
            F = np.empty(lambda_)
            for i in range(lambda_):
                F[i] = func(np.clip(X[i], -5, 5))
            calls += lambda_

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_fx_cma:
                best_fx_cma = F_sorted[0]
                best_x_cma = X_sorted[0].copy()

            # Mean update
            m_new = X_sorted[:mu] @ weights[:, None]

            # Adaptation of step size
            y_old = C_inv_sqrt @ (m_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) \
                   < (1 + 2 / (n + 1)) * chi_n
            if hsig:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Covariance matrix adaptation
            if hsig:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma
            else:
                p_c = (1 - c_c) * p_c
            y_c = p_c / np.linalg.norm(p_c) if np.linalg.norm(p_c) > 1e-16 else np.zeros(n)
            rank_c = (1 - c_c) * (1 + c_c * (2 - hsig)) * np.outer(y_c, y_c)

            # Rank-mu update
            diff = (X_sorted[:mu] - x[:, None]) / sigma
            rank_mu = c_mu * (weights[:, None] * diff @ diff.T).T  # weights[i] * diff[i] @ diff[i].T
            # More precisely:
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(diff[i], diff[i])
            rank_mu *= c_mu

            C = (1 + c_1 * (rank_c - C) + c_mu * rank_mu) * C
            # Ensure positive definite
            C = (C + C.T) / 2
            # Re-regularize
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_c = np.maximum(eigvals_c, 1e-12)
            C = (eigvecs_c * eigvals_c) @ eigvecs_c.T
            C_inv_sqrt = (eigvecs_c * (1.0 / np.sqrt(eigvals_c))) @ eigvecs_c.T

            x = m_new
            sigma = max(sigma, 1e-10)

            # If we've made good progress, continue; otherwise restart might help
            if gen % 50 == 0 and gen > 0:
                # Periodic restart if stuck
                if F_sorted[0] >= best_fx_cma * 1.001 and np.allclose(x, best_x_cma, atol=1e-4):
                    # Try a small perturbation
                    x = best_x_cma + sigma * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)

        # Phase 3: Fine local search with (1+1)-ES from best point
        x_local = best_x_cma.copy()
        fx_local = best_fx_cma
        sigma_local = 0.5
        remaining_local = budget - calls
        for _ in range(min(remaining_local, max(10, budget // 10))):
            if not time_ok():
                break
            y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx_local:
                x_local, fx_local = y, fy
                sigma_local *= 1.3
            else:
                sigma_local *= 0.75

            if fx_local < best_fx_cma:
                best_fx_cma = fx_local
                best_x_cma = x_local.copy()

            if sigma_local < 1e-8:
                break

        return best_x_cma, best_fx_cma
# EVOLVE-BLOCK-END
