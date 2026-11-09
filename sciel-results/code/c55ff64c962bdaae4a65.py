"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size and restart rule for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def cmaes(x_init, sigma_init, lambda_pop):
            nonlocal calls_made, best_x, best_f

            mu = max(2, lambda_pop // 2)
            weights = np.ones(mu)
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            x_mean = x_init.copy()
            f_mean = func(x_mean)
            calls_made += 1
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            gen = 0
            no_improve = 0

            while calls_made < budget:
                # Generate population
                B = rng.standard_normal((lambda_pop, dim))
                try:
                    D = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    D = np.eye(dim)

                pop = np.array([x_mean + sigma * (B[i] @ D) for i in range(lambda_pop)])
                pop = np.clip(pop, -5, 5)

                # Check if we have enough budget
                if calls_made + lambda_pop > budget:
                    n_eval = budget - calls_made
                    fvals = np.array([func(pop[i]) for i in range(n_eval)])
                    calls_made += n_eval
                    fvals = np.concatenate([fvals, np.full(lambda_pop - n_eval, np.inf)])
                    lambda_pop_eff = n_eval
                else:
                    fvals = np.array([func(pop[i]) for i in range(lambda_pop)])
                    calls_made += lambda_pop
                    lambda_pop_eff = lambda_pop

                # Update best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < best_f:
                    best_f = fvals[idx_best]
                    best_x = pop[idx_best].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Sort by fitness
                order = np.argsort(fvals)[:min(mu, lambda_pop_eff)]
                x_sorted = pop[order]

                # Update mean
                y_w = np.zeros(dim)
                for i in range(len(order)):
                    y_w += weights[i] * (x_sorted[i] - x_mean) / sigma
                x_old = x_mean.copy()
                x_mean = x_mean + sigma * y_w
                x_mean = np.clip(x_mean, -5, 5)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_w
                ps_norm = np.linalg.norm(ps)
                hs = ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / chi_n
                if hs < 1.4 + 2 / (dim + 1):
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_w
                else:
                    pc = (1 - c_c) * pc

                # Update covariance matrix
                delta_hs = (1 - hs) * c_c * (2 - c_c)
                delta_mu = (1 - delta_hs) * c_mu

                # Rank-1 update
                C = C * (1 - delta_hs - delta_mu) + np.outer(ps, ps) * (1 - delta_hs) * (c_c * (2 - c_c) * mu_eff - 1)

                # Rank-mu update
                for i in range(len(order)):
                    diff = (x_sorted[i] - x_old) / sigma
                    C += weights[i] * np.outer(diff, diff) * delta_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / damps * (ps_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                gen += 1

                # Restart criteria
                if no_improve > 40:
                    break
                if sigma < 1e-6 or sigma > 1e8:
                    break
                if calls_made >= budget:
                    break

            return calls_made

        # Phase 1: Global exploration with diverse starts
        # Use a few well-spread initial points
        initial_points = [
            np.zeros(dim),
            np.full(dim, 4.209687),  # Schwefel's known minimum (unshifted)
            np.full(dim, -4.209687),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        
        # Shuffle to avoid bias
        rng.shuffle(initial_points)
        
        for x_init in initial_points:
            if calls_made >= budget:
                break
            # Start with large sigma for global exploration
            sigma_init = 3.0
            lambda_pop = 6
            cmaes(x_init, sigma_init, lambda_pop)
        
        # Phase 2: Local refinement around best point
        if calls_made < budget and best_x is not None:
            # Use smaller population for fine-tuning
            remaining = budget - calls_made
            if remaining > 20:
                cmaes(best_x.copy(), 0.5, 4)
            
            # Phase 3: Very local refinement
            if calls_made < budget and best_x is not None:
                remaining = budget - calls_made
                if remaining > 10:
                    cmaes(best_x.copy(), 0.1, 3)

        return best_x, best_f
# EVOLVE-BLOCK-END
