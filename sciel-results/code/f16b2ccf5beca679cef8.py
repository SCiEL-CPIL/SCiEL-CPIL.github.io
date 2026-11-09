"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        n_calls = 1

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda
        sigma = 1.0
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = (2 + 2 / (dim + 1) - c1) / ((dim + 2) ** 2 + 5)
        cs = (dim + 1.3) / (dim + 5 + cs if False else dim + 5)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        c_sigma = cs / damps
        cc = (4 + dim / dim) / (dim + 4)
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = (2 + 2 / (dim + 1) - c1) / ((dim + 2) ** 2 + 5)

        # Covariance matrix
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # inverse of square root
        mean = x_best.copy()

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Step size adaptation (1/5 rule hybrid with CMA-ES)
        sigma = 1.5

        for gen in range(budget // pop_size + 1):
            if n_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if n_calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Use simple isotropic step with adaptive sigma
                y = mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                n_calls += 1
                offspring.append((fy, y))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < f_best:
                f_best = offspring[0][0]
                x_best = offspring[0][1].copy()

            # Selection
            mu = max(1, pop_size // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)

            # Weighted mean of selected
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += w[i] * offspring[i][1]

            # Update mean
            mean = y_mean

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (y_mean - mean + sigma * np.zeros(dim)) / sigma if False else (1 - c_sigma) * p_sigma
            # Simplified: just use mean shift
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_mean - mean) / sigma if sigma > 1e-10 else p_sigma
            # Actually, let's keep it simpler for robustness

            # Adaptive sigma using success rate
            success_count = 0
            for i in range(mu):
                if offspring[i][0] < f_best + 1e-10:
                    success_count += 1
            # Better: count how many improved over the mean
            prev_mean_fitness = None  # We don't track this easily, use simple rule
            # Use 1/5 rule: if more than 1/5 of offspring are better than previous best, increase
            # Simpler: track if best improved this generation
            # Let's use a simpler adaptive strategy

            # Simple sigma adaptation: if best improved, increase slightly; else decrease
            # This is more like (1+1)-ES but with population

            # Let's just use a fixed schedule mixed with success
            if offspring[0][0] < f_best:
                sigma *= 1.1
            else:
                sigma *= 0.95

            sigma = np.clip(sigma, 0.01, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
