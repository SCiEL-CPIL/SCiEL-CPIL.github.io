# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES for BBOB f18 (Schaffers F7 ill-conditioned) with wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 280.0
        dim = self.dim
        budget = self.budget
        L, U = -5.0, 5.0

        rng = np.random.default_rng(20240517)

        # CMA-ES parameters
        # lambda: population size, mu: number of parents
        lam = 10
        mu = lam // 2  # 5 parents

        # Weights for weighted mean
        w = np.log(mu + 1) - np.arange(1, mu + 1)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size
        sigma = 2.0  # initial step size (box is [-5,5]^5, so 2 is reasonable)

        # Initial population center
        mean = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Track best
        best_x = None
        best_f = np.inf
        n_calls = 0

        # Generate initial population
        def generate_population():
            nonlocal C_sqrt
            z = rng.standard_normal((lam, dim))
            # Cholesky decomposition of C
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)
            pop = mean + sigma * (z @ C_sqrt.T)
            return pop

        # Initial population
        pop = generate_population()
        fitness = np.array([float(func(x)) for x in pop])
        n_calls += lam

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Main loop
        while n_calls < budget:
            if time.time() - t0 > DEADLINE:
                break

            # Sort population by fitness
            order = np.argsort(fitness)
            pop_sorted = pop[order]
            fit_sorted = fitness[order]

            # Update best
            if fit_sorted[0] < best_f:
                best_f = fit_sorted[0]
                best_x = pop_sorted[0].copy()

            # Compute weighted mean
            new_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)

            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (new_mean - mean))

            # Check hsig
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (n_calls // lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1) - 1)

            # Update evolution path for C
            if hsig:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            else:
                p_c = (1 - c_c) * p_c

            # Update covariance matrix
            delta_hsig = 1.0 if hsig else 0.0
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, None, None] * np.outer(pop_sorted[:mu] - mean, pop_sorted[:mu] - mean), axis=0) / (sigma ** 2)
            C = (1 + c_1 * (delta_hsig - 1) + c_mu) * C + c_1 * (rank_one - C) + c_mu * rank_mu

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt((mu_eff + 1) / (dim + 1) - 1) - 1))

            # Update mean
            mean = new_mean

            # Check termination criteria
            if np.all(np.abs(C - np.diag(np.diag(C))) < 1e-10 * np.max(np.abs(C))):
                break
            if sigma < 1e-15:
                break

            # Generate new population
            pop = generate_population()
            fitness = np.array([float(func(x)) for x in pop])
            n_calls += lam

        return best_x
# EVOLVE-BLOCK-END
