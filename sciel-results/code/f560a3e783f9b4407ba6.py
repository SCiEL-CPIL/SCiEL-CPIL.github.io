"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_pop // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        def get_cholesky(C):
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-12)
                L = evecs @ np.diag(np.sqrt(evals))
            return L

        C_sqrt = get_cholesky(C)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial population
        for i in range(lambda_pop):
            z = rng.standard_normal(d)
            x_i = x_mean + sigma * (C_sqrt @ z)
            evaluate(x_i)

        # Main loop
        while calls_made < budget:
            # Generate population
            offspring = []
            fitnesses = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(d)
                x_i = x_mean + sigma * (C_sqrt @ z)
                f_i = evaluate(x_i)
                offspring.append(x_i)
                fitnesses.append(f_i)

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            idx = np.argsort(fitnesses)
            offspring = offspring[idx]
            fitnesses = fitnesses[idx]

            # Recombination
            x_new = np.sum(offspring[:mu] * weights[:, None], axis=0)
            x_mean_old = x_mean.copy()
            x_mean = x_new

            # Update evolution paths
            d_x = (x_mean - x_mean_old) / sigma
            C_inv_sqrt = np.linalg.inv(C_sqrt).T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (4 - c_sigma) * mu_eff) * (C_inv_sqrt @ d_x)
            norm_ps = np.linalg.norm(ps)

            # Update pc
            h_sigma = 1.0 if norm_ps / chi_n < 1.4 + 2 / (d + 1) else 0.0
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (4 - c_c) * mu_eff) * d_x

            # Update covariance matrix
            outer_pc = np.outer(pc, pc)
            # Rank-mu update
            z_offspring = (offspring[:mu] - x_mean_old) / sigma
            outer_mu = np.sum(weights[:, None, None] * z_offspring[:, :, None] * z_offspring[:, None, :], axis=0)
            
            C = (1 + c_1 * h_sigma * (c_c * (4 - c_c) - 1) + c_mu - c_1) * C \
                + c_1 * (1 + h_sigma * (c_c * (4 - c_c) - 1) - c_mu) * outer_pc \
                + c_mu * outer_mu

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure positive definiteness and update Cholesky
            C = (C + C.T) / 2
            C_sqrt = get_cholesky(C)

        return best_x, best_f
# EVOLVE-BLOCK-END
