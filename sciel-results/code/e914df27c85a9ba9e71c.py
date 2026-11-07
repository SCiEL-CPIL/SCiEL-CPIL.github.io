"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart strategy and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def eval_point(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-restart CMA-ES
        # Each restart gets a portion of the budget
        num_restarts = 4
        budget_per_restart = max(50, budget // num_restarts)

        for restart in range(num_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize center
            if restart == 0:
                center = np.zeros(dim)
            else:
                # Try different initialization strategies
                if restart == 1:
                    center = np.ones(dim) * 2.0
                elif restart == 2:
                    center = -np.ones(dim) * 2.0
                else:
                    rng_init = np.random.default_rng(restart * 42 + 7)
                    center = rng_init.uniform(-3, 3, dim)

            # CMA-ES parameters
            n = dim
            lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lambda_ // 2  # number of parents
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size
            sigma = 1.0

            # Covariance matrix
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)

            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 - mu_eff / (n + 1) ** 2 + 2 * mu_eff / n) / (
                (n + 2) ** 2 + mu_eff)
            c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / (
                (n + 2) ** 2 + mu_eff)

            # Track best in this restart
            restart_best_f = float('inf')
            restart_best_x = center.copy()

            # Initial population
            pop = []
            for i in range(lambda_):
                x_i = center + sigma * np.random.standard_normal(n)
                x_i = np.clip(x_i, -5, 5)
                fx_i = eval_point(x_i)
                if fx_i is None:
                    break
                pop.append((fx_i, x_i))
                if fx_i < restart_best_f:
                    restart_best_f = fx_i
                    restart_best_x = x_i.copy()

            if len(pop) < 2:
                continue

            # Main loop
            max_iters = budget_per_restart
            for it in range(max_iters):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[0])

                # Select parents
                parents = [p[1] for p in pop[:mu]]
                parents = np.array(parents)

                # Weighted mean
                mean = np.sum(weights[:, None] * parents, axis=0)

                # Update evolution paths
                old_mean = center
                center = mean

                # Update sigma evolution path
                if np.linalg.norm(center - old_mean) > 0:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                        c_sigma * (2 - c_sigma) * mu_eff) * (
                        center - old_mean) / (sigma * np.linalg.norm(
                            center - old_mean))

                # Update covariance evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(
                    c_c * (2 - c_c) * mu_eff) * (center - old_mean) / sigma

                # Update covariance matrix
                delta_d = (1 + c_sigma * (2 - c_sigma)) * np.dot(
                    p_sigma, p_sigma) - (1 - c_sigma)
                rank1 = (1 - c_1) * np.outer(p_c, p_c) + c_1 * (
                    (1 + (1 - c_c) * delta_d) * np.outer(p_c, p_c) - np.eye(n))

                # Rank-mu update
                diff = (parents - old_mean) / sigma
                rank_mu = c_1 * np.eye(n) + c_mu * np.sum(
                    [weights[i] * np.outer(diff[i], diff[i]) for i in range(mu)],
                    axis=0)

                C = (1 - c_1 - c_mu) * C + rank1 + rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (
                    np.linalg.norm(p_sigma) / np.sqrt(2 * n) * (1 + 1e-9) - 1))

                # Clip sigma to reasonable range
                sigma = np.clip(sigma, 1e-6, 10.0)

                # Generate new population
                new_pop = []
                for i in range(lambda_):
                    z = np.random.standard_normal(n)
                    x_i = center + sigma * z
                    x_i = np.clip(x_i, -5, 5)
                    fx_i = eval_point(x_i)
                    if fx_i is None:
                        break
                    new_pop.append((fx_i, x_i))
                    if fx_i < restart_best_f:
                        restart_best_f = fx_i
                        restart_best_x = x_i.copy()

                if len(new_pop) < 2:
                    break
                pop = new_pop

            # Update global best
            if restart_best_f < best_f:
                best_f = restart_best_f
                best_x = restart_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
