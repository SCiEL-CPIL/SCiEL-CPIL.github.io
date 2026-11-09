"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with initial diverse sampling for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Diverse initial sampling (12 points)
        # Include center and structured points for fast initial progress
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            initial_points.append(p)
            q = np.zeros(dim)
            q[i] = -2.0
            initial_points.append(q)
        # A couple random
        while len(initial_points) < 12:
            initial_points.append(rng.uniform(-3, 3, dim))

        best_x = None
        best_f = np.inf
        calls = 0

        for p in initial_points:
            if calls >= budget:
                break
            p_clipped = np.clip(p, -5, 5)
            f = func(p_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p_clipped.copy()

        # Phase 2: CMA-ES
        if calls >= budget or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 4  # lambda
        sigma = 1.5
        x_mean = best_x.copy()
        C = np.eye(dim)  # covariance
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Precompute Cholesky of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        for gen in range(200):
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            # Check budget
            if calls + pop_size > budget:
                # Do partial generation
                remaining = budget - calls
                if remaining <= 0:
                    break
                # Generate only remaining points
                for _ in range(remaining):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                break

            # Generate population
            offspring = []
            fitnesses = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fitnesses.append(func(y))
                calls += 1

            # Sort by fitness
            order = np.argsort(fitnesses)
            offspring = [offspring[i] for i in order]
            fitnesses = [fitnesses[i] for i in order]

            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = offspring[0].copy()

            # Weighted mean of top mu
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i]
            x_mean = new_mean

            # Update sigma and C
            y_vec = (new_mean - x_mean) / (sigma * np.sqrt(mu_eff)) if sigma > 1e-12 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - x_mean) / sigma if sigma > 1e-12 else (1 - c_c) * p_c

            # Rank-one update
            C = (1 + c_1 * (mu_eff - 1) * (c_c * (2 - c_c) + 1) / (1 - c_c) - c_1) * C \
                + c_1 * np.outer(new_mean - x_mean, new_mean - x_mean) / sigma**2 if sigma > 1e-12 else C

            # Rank-mu update
            for i in range(mu):
                z_i = (offspring[i] - x_mean) / sigma if sigma > 1e-12 else np.zeros(dim)
                C += c_mu * weights[i] * np.outer(z_i, z_i)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim * (1 - np.exp(-2 * c_sigma))) - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Check if we've converged
            if sigma < 1e-6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
