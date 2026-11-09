"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 1.8 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff / 2 - 1) + 2 / (n + 2)) /
                  ((n + 2) ** 1.8 + mu_eff))

        def run_cmaes(x0, sigma0, budget_remaining):
            """Run CMA-ES for budget_remaining evaluations."""
            if budget_remaining <= 0:
                return x0, float('inf')

            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Covariance matrix
            C = np.eye(n)
            # Precompute Cholesky factor
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(n)
                C_sqrt = np.eye(n)

            best_x = x.copy()
            best_f = float('inf')
            calls_made = 0

            # Initial evaluation
            f_best = func(x)
            calls_made += 1
            if f_best < best_f:
                best_f = f_best
                best_x = x.copy()

            if calls_made >= budget_remaining:
                return best_x, best_f

            while calls_made < budget_remaining:
                # Generate population
                ys = np.empty((lambda_, n))
                fs = np.empty(lambda_)
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    # Transform using Cholesky
                    try:
                        y = x + sigma * (C_sqrt @ z)
                    except Exception:
                        y = x + sigma * z
                    ys[i] = np.clip(y, -5, 5)
                    fs[i] = func(ys[i])
                    calls_made += 1
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = ys[i].copy()
                    if calls_made >= budget_remaining:
                        break

                if calls_made >= budget_remaining:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                y_selected = ys[order[:mu]]
                f_selected = fs[order[:mu]]

                # Weighted mean of selected
                m = np.sum(weights[:, None] * y_selected, axis=0)

                # Update evolution paths
                try:
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m - x) / sigma
                h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_)))) < \
                            (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_))) else 0.0

                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

                # Update covariance matrix
                delta_d = (1 - h_sigma) * c_c * (2 - c_c) + h_sigma * c_mu
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(weights[:, None, None] * np.outer(y_selected - x, y_selected - x), axis=0) / sigma ** 2

                try:
                    C = (1 + c_1 * (delta_d - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                    # Ensure symmetry
                    C = (C + C.T) / 2
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_))) - np.sqrt(n) * (1 - np.exp(-0.05 * (1 - c_sigma)))))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update mean
                x = m

                # Check for stagnation
                if sigma < 1e-10:
                    break

            return best_x, best_f

        # Multi-restart strategy
        # First run: standard CMA-ES with a reasonable initial point
        x0 = np.zeros(n)
        # Try a few random initializations to find a good starting point
        best_x_overall = np.zeros(n)
        best_f_overall = float('inf')

        # Quick random search to find a good starting point
        n_init = min(10, budget // 10)
        for i in range(n_init):
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            if f_try < best_f_overall:
                best_f_overall = f_try
                best_x_overall = x_try.copy()
            budget_remaining = budget - (i + 1)
            if budget_remaining <= 10:
                break

        # Run CMA-ES from the best point found
        budget_remaining = budget - n_init
        if budget_remaining > 10:
            x_best, f_best = run_cmaes(best_x_overall.copy(), 1.0, budget_remaining)
            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best.copy()

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
