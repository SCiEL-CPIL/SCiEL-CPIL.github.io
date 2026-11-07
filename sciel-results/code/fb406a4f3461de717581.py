"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # We will run a CMA-ES-like algorithm with population size lambda
        # Adaptively restart if no improvement for a while

        def run_cmaes(start_x, budget_remaining):
            nonlocal calls_used, best_x, best_f

            if budget_remaining <= 0:
                return

            # Population size
            lam = max(4, int(4 + 3 * dim))  # standard CMA-ES lambda
            mu = lam // 2

            # Initialize
            x_mean = np.array(start_x, dtype=float)
            sigma = 1.0

            # CMA-ES parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Compute mu_eff
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Covariance matrix
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C_inv_sqrt = np.eye(dim)

            # Initial population
            ys = np.array([rng.standard_normal(dim) for _ in range(lam)])
            xs = np.array([x_mean + sigma * y for y in ys])
            xs = np.clip(xs, -5, 5)

            # Evaluate initial population
            fs = np.array([func(x) for x in xs])
            calls_used += lam

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Weighted mean
            w = weights[:mu]
            x_new_mean = np.sum(w[:, None] * xs[:mu], axis=0)

            # Update path
            C_inv_sqrt_ys = C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_ys
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_used / dim + 1)) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - c_c) * pc + hs * c_c * (x_new_mean - x_mean) / sigma

            # Update covariance
            delta_hs = (1.0 if hs else 0.0) - (1 - c_c)
            rank_mu = np.sum(w[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Add regularization
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

            x_mean = x_new_mean
            budget_remaining -= lam

            # Check for convergence
            if np.linalg.norm(x_new_mean - x_mean) < 1e-10 or sigma < 1e-10:
                return

            # Continue with more iterations if budget allows
            while budget_remaining >= lam:
                ys = np.array([C_inv_sqrt @ rng.standard_normal(dim) for _ in range(lam)])
                xs = np.array([x_mean + sigma * y for y in ys])
                xs = np.clip(xs, -5, 5)

                fs = np.array([func(x) for x in xs])
                calls_used += lam
                budget_remaining -= lam

                order = np.argsort(fs)
                ys = ys[order]
                xs = xs[order]
                fs = fs[order]

                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()

                x_new_mean = np.sum(w[:, None] * xs[:mu], axis=0)

                C_inv_sqrt_ys = C_inv_sqrt @ (x_new_mean - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_ys
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_used / dim + 1)) / chi_n < 1.4 + 2 / (dim + 1)
                pc = (1 - c_c) * pc + hs * c_c * (x_new_mean - x_mean) / sigma

                delta_hs = (1.0 if hs else 0.0) - (1 - c_c)
                rank_mu = np.sum(w[:, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.inv(L)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                        C_inv_sqrt = np.linalg.inv(L)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        C_inv_sqrt = np.eye(dim)

                x_mean = x_new_mean

                if np.linalg.norm(x_new_mean - x_mean) < 1e-10 or sigma < 1e-10:
                    break

        # Run CMA-ES from a few different starting points, splitting budget
        n_starts = 3
        budget_per_start = total_budget // n_starts

        for i in range(n_starts):
            if calls_used >= total_budget:
                break
            # Different starting points
            if i == 0:
                start = np.zeros(dim)
            elif i == 1:
                start = rng.uniform(-3, 3, dim)
            else:
                start = rng.uniform(-4, 4, dim)

            run_cmaes(start, min(budget_per_start, total_budget - calls_used))

        return best_x, best_f

# EVOLVE-BLOCK-END
