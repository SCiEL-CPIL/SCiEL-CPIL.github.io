# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based CMA-ES with multi-restart for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(dim)
            f = float(func(x))
            calls_made += 1
            return f

        # Strategy: 3 restarts to escape local minima
        # Each restart uses CMA-ES with population-based sampling
        num_restarts = 3
        budget_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline or calls_made >= budget:
                break

            # Initialize center
            if restart == 0:
                center = rng.uniform(-4, 4, dim)
            else:
                # Start near previous best with some noise
                center = best_x + rng.normal(0, 1.0, dim)
                center = np.clip(center, -4.5, 4.5)

            # CMA-ES parameters
            mu = dim // 2  # number of parents
            lambda_ = 4 * dim  # population size
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            # Step size
            sigma = 1.5 / (np.sqrt(2 * dim) * (1 + 0.5 * np.log(dim)))
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_C = np.zeros(dim)
            # Damping
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Generation loop
            max_gen = budget_per_restart // lambda_
            for gen in range(max_gen):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Generate population
                offspring = []
                offspring_f = []

                for i in range(lambda_):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    # Sample from multivariate normal using C
                    z = rng.standard_normal(dim)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    except np.linalg.LinAlgError:
                        L = np.linalg.cholesky(C + 1e-6 * np.eye(dim))
                    y = center + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) == 0:
                    break

                # Sort offspring
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]

                # Update best
                if offspring_f[0] < best_f:
                    best_f = offspring_f[0]
                    best_x = offspring[0].copy()

                # Update mean
                new_center = sum(w[i] * offspring[i] for i in range(mu))

                # Update sigma
                y_mean = (new_center - center) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
                sigma_factor = np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
                sigma = np.clip(sigma * sigma_factor, 1e-10, 10)

                # Update covariance
                y_w = (new_center - center)
                rank_mu_term = sum(w[i] * np.outer(offspring[i] - center, offspring[i] - center) for i in range(mu))
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(y_w, y_w) + (1 - (1 - c_1 - c_mu) * (1 - (1 - c_1) / (c_1 * dim))) * np.eye(dim)) + c_mu * rank_mu_term

                # Update evolution path for covariance
                p_C = (1 - c_c) * p_C + c_c * (np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma) * (new_center - center)

                center = new_center

                # Check convergence
                if sigma < 1e-15:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
