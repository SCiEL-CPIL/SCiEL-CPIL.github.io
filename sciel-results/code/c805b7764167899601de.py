"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and adaptive restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Global best across all restarts
        global_best_x = None
        global_best_f = np.inf

        # We'll do multiple short CMA-ES runs with different initializations
        # Split budget into chunks for restarts
        n_restarts = 5
        budget_per_restart = max(1, budget // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize with different strategies per restart
            if restart == 0:
                # Start near origin
                x0 = np.zeros(n)
            elif restart == 1:
                # Start at random point
                x0 = np.random.uniform(-4, 4, n)
            elif restart == 2:
                # Start at another random point
                x0 = np.random.uniform(-3, 3, n)
            elif restart == 3:
                # Start at origin with small perturbation
                x0 = np.random.normal(0, 0.5, n)
            else:
                # Start at a random point biased toward origin
                x0 = np.random.uniform(-2, 2, n)

            # Clip to box
            x0 = np.clip(x0, -5, 5)

            # CMA-ES parameters
            lambda_ = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            mu = lambda_ // 2
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            # Step size
            sigma = 2.0 if restart == 0 else 1.5

            # Mean
            mean = x0.copy()

            # Covariance matrix
            C = np.eye(n)

            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)

            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

            # Hessian approximation update
            hsig = np.sqrt((1 - c_c) / (1 + c_c)) * (2 + c_mu / c_1)

            calls_used = 0
            stall_count = 0
            best_f_in_run = np.inf
            best_x_in_run = None

            for gen in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if calls_used >= budget_per_restart:
                    break

                # Generate population
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(n)
                    L = np.eye(n)

                # Generate lambda_ candidates
                candidates = []
                for i in range(lambda_):
                    z = np.random.standard_normal(n)
                    y = mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    candidates.append(y)

                # Evaluate all candidates
                fvals = np.array([func(y) for y in candidates])
                calls_used += lambda_

                # Sort by fitness
                idx = np.argsort(fvals)
                sorted_fvals = fvals[idx]
                sorted_candidates = [candidates[i] for i in idx]

                # Update best
                if sorted_fvals[0] < best_f_in_run:
                    best_f_in_run = sorted_fvals[0]
                    best_x_in_run = sorted_candidates[0].copy()

                if best_f_in_run < global_best_f:
                    global_best_f = best_f_in_run
                    global_best_x = best_x_in_run.copy()

                # Check for stall
                if sorted_fvals[0] < best_f_in_run - 1e-10:
                    stall_count = 0
                else:
                    stall_count += 1

                # If stalled too long, break out of this restart
                if stall_count > 30:
                    break

                # Update mean
                new_mean = np.sum([weights[i] * sorted_candidates[i] for i in range(mu)], axis=0)
                mean = new_mean
                mean = np.clip(mean, -5, 5)

                # Update evolution paths
                # Standardized mean
                if sigma > 1e-15:
                    # Need to compute C^{-1/2}
                    try:
                        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                    except np.linalg.LinAlgError:
                        C_inv_sqrt = np.eye(n)

                    y_mean = (new_mean - mean) / sigma if sigma > 0 else np.zeros(n)
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_mean
                    norm_ps = np.linalg.norm(ps)
                    hsig = 1.0  # Simplified
                    if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) * (1 - (1 - c_sigma)**(2 * (gen + 1)))):
                        pc = (1 - c_c) * pc + c_c * C_inv_sqrt @ y_mean
                    else:
                        pc = (1 - c_c) * pc

                    # Update covariance
                    delta_h = (1 + hsig * (norm_ps**2 / n - 1)) * c_c * (2 - c_c)
                    outer = np.outer(pc, pc)
                    C = (1 + c_1 + c_mu - c_1 - delta_h) * C
                    C += c_1 * (outer - C)
                    # Rank-mu update
                    for i in range(mu):
                        y_i = (sorted_candidates[i] - mean) / sigma
                        C += c_mu * np.outer(y_i, y_i)
                else:
                    C = np.eye(n)

                # Update sigma
                if norm_ps > 0:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n) - 1))
                else:
                    sigma *= np.exp(-c_sigma / d_sigma)

                # Clip sigma
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # If C has become ill-conditioned, reset
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if eigvals.min() < 1e-10 or eigvals.max() / max(eigvals.min(), 1e-10) > 1e10:
                        C = np.eye(n)
                        sigma = 1.0
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    sigma = 1.0

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
