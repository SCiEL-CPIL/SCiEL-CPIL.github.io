"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Two-phase optimizer: coarse random/grid search followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Phase 1: Exploration
        best_x = None
        best_f = np.inf
        calls_made = 0

        # 1. Center point
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_f = f0
        best_x = x0.copy()

        # 2. Structured exploration: sample along axes and corners
        # Sample at multiple radii along each axis
        axis_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        for i in range(dim):
            for val in axis_values:
                for sign in [1, -1]:
                    x = np.zeros(dim)
                    x[i] = sign * val
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # 3. Random points with multiple distributions
        n_explore_target = max(80, min(150, budget // 4))
        n_random = n_explore_target - calls_made

        if n_random > 0:
            for _ in range(n_random):
                # Mix of uniform and Gaussian samples
                if rng.random() < 0.5:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = rng.normal(0, 2.0, dim)
                    x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: CMA-ES with tuned parameters
        # Use larger population for better exploration
        lambda_pop = max(10, 4 + int(np.floor(3 * np.log(dim))))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Standard CMA-ES parameters
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
        cmu = (2 + 2 / (dim + 1) - dim * c1) / ((dim + 1.3) ** 2 + 2)
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mean = best_x.copy()
        C = np.eye(dim)
        # Initial sigma: estimate from spread of good points
        sigma = 0.5
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        gen = 0
        last_improvement_gen = 0
        best_f_at_last_improvement = best_f

        # Main CMA-ES Loop
        while calls_made + lambda_pop <= budget:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)

            offspring = np.empty((lambda_pop, dim))
            offspring_f = np.empty(lambda_pop)

            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                offspring[i] = y
                offspring_f[i] = func(y)
                calls_made += 1

                # Update global best
                if offspring_f[i] < best_f:
                    best_f = offspring_f[i]
                    best_x = y.copy()
                    last_improvement_gen = gen
                    best_f_at_last_improvement = best_f

            # Sort offspring by fitness
            sorted_idx = np.argsort(offspring_f)
            sorted_offspring = offspring[sorted_idx]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_offspring[i]

            # Update step size control (ps)
            diff_mean = new_mean - mean
            try:
                diff_scaled = np.linalg.solve(L.T, diff_mean / sigma)
            except np.linalg.LinAlgError:
                diff_scaled = diff_mean / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * diff_scaled

            # Update covariance control (pc)
            denom = np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            if denom < 1e-10:
                denom = 1e-10
            hsig = (np.linalg.norm(ps) / denom) < (1.4 + 2 / (dim + 1)) * chi_n

            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * diff_mean / sigma

            # Update covariance matrix
            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * np.outer(ps, ps))

            for i in range(mu):
                y_i = (sorted_offspring[i] - mean) / sigma
                C_new += cmu * weights[i] * np.outer(y_i, y_i)

            C = C_new
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            gen += 1

            # Early stopping: if no improvement for many generations and sigma is small
            if gen - last_improvement_gen > 15 and sigma < 0.01:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
