"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Track global best
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        # lambda: population size
        # For dim=5, use lambda = 4 + floor(3*ln(n)) = 4 + floor(3*1.609) = 4+4 = 8
        lam = 4 + int(3 * np.log(n))
        # We'll use a slightly larger population for better exploration
        lam = max(lam, 10)

        # We divide the budget among restarts
        # Each restart gets a portion of the budget
        # Let's use adaptive restarts based on stagnation

        # Initial sigma (step size)
        # The search space is [-5,5]^5, so initial spread of ~2-3 is reasonable
        sigma0 = 2.0

        # Number of restarts: we'll allocate budget per restart
        # Each restart gets min(budget_per_restart, remaining) evaluations
        # Let's use a base budget per restart and increase if we have time
        base_budget_per_restart = 80
        max_restarts = 6

        for restart in range(max_restarts):
            if time.time() >= deadline:
                break

            # Remaining budget
            remaining = budget - (restart * base_budget_per_restart)
            if remaining <= 0:
                break

            # Budget for this restart
            budget_this = min(base_budget_per_restart, remaining)
            if budget_this < lam + 1:
                break

            # Initialize CMA-ES
            # Start from origin or from previous best if available
            if restart == 0:
                x_mean = np.zeros(n)
            else:
                # Perturb the previous best slightly
                if global_best_x is not None:
                    rng_restart = np.random.default_rng(42 + restart)
                    x_mean = np.clip(global_best_x + rng_restart.normal(0, 0.5, n), -5, 5)
                else:
                    x_mean = np.zeros(n)

            sigma = sigma0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # CMA-ES update parameters
            mu = lam // 2  # number of weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((n + 2) ** 2 + mu_eff))

            # Evaluate initial population
            rng = np.random.default_rng(42 + restart)
            pop = x_mean + sigma * rng.multivariate_normal(np.zeros(n), C, size=lam)
            pop = np.clip(pop, -5, 5)
            fvals = np.array([func(x) for x in pop])

            # Update global best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < global_best_f:
                global_best_f = fvals[idx_best]
                global_best_x = pop[idx_best].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Main CMA-ES loop
            for gen in range(1, budget_this - lam):
                if time.time() >= deadline:
                    return global_best_x, global_best_f

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

                # Update evolution paths
                z = np.linalg.solve(np.linalg.cholesky(C).T, (x_mean - x_mean_old) / sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / np.sqrt(mu_eff) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))

                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * z

                # Update covariance matrix
                delta_h = (1 - h_sig) * c_c * (2 - c_c) * np.linalg.norm(p_c) ** 2 / n
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                    + c_1 * np.outer(p_c, p_c) \
                    + c_mu * np.sum(weights[:mu] * np.outer(pop_sorted[:mu] - x_mean_old, pop_sorted[:mu] - x_mean_old) / sigma ** 2, axis=0)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

                # Ensure sigma stays in reasonable bounds
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Generate new population
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset covariance
                    C = np.eye(n)
                    L = np.eye(n)

                pop = x_mean + sigma * (L @ rng.standard_normal((lam, n))).T
                pop = np.clip(pop, -5, 5)
                fvals = np.array([func(x) for x in pop])

                # Update global best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < global_best_f:
                    global_best_f = fvals[idx_best]
                    global_best_x = pop[idx_best].copy()

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Check for stagnation: if no improvement in best value for many generations
                # This is a simple check; we could be more sophisticated

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
