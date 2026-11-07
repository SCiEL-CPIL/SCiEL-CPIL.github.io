"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f5 (linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = int(4 + 3 * np.log(dim))  # For dim=5: 4+3*1.609 ≈ 8
        pop_size = max(4, pop_size)

        # Initialize
        x = np.zeros(dim)  # Start at center
        sigma = 2.0  # Initial step size
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # CMA-ES update parameters
        mu = pop_size // 2  # Number of selected solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 - mueff / mueff + 3) / (dim + 4 - mueff / mueff + 3)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        evals = 0
        best_x = x.copy()
        best_f = np.inf

        def time_check():
            return time.time() > deadline

        # Evaluate initial point
        fx = func(x)
        evals += 1
        best_f = fx
        best_x = x.copy()

        for gen in range(budget):
            if time_check():
                break

            # Generate population
            population = []
            f_values = []

            for i in range(pop_size):
                if time_check():
                    break

                # Standard normal
                z = np.random.randn(dim)
                # Transform using C^0.5
                x_trial = x + sigma * (C_sqrt @ z)
                x_trial = np.clip(x_trial, -5, 5)

                fx_trial = func(x_trial)
                evals += 1

                population.append(x_trial)
                f_values.append(fx_trial)

                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()

            if len(population) < 2:
                break

            # Sort by fitness
            order = np.argsort(f_values)
            population = [population[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Update mean
            x_new = np.sum([weights[i] * population[i] for i in range(mu)], axis=0)

            # Update sigma
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ (x_new - x) / sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = max(1e-10, min(sigma, 100.0))

            # Update C
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * C_sqrt_inv @ (x_new - x) / sigma

            # Rank-one and rank-mu updates
            v = p_c / chi_N
            C = (1 - c_1 - c_mu) * C + (c_1 + c_h) * np.outer(v, v) + c_1 * np.outer(p_c, p_c) + \
                c_mu * np.sum([weights[i] * np.outer(C_sqrt_inv @ (population[i] - x) / sigma,
                                                      C_sqrt_inv @ (population[i] - x) / sigma)
                               for i in range(mu)], axis=0)

            # c_h correction
            c_h = c_c * (1 + 2 * min((mueff - 1) / (dim + 1) - 1, 0) + c_sigma)

            # Recompute C^0.5
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_sqrt = np.eye(dim)

            x = x_new

            if evals >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
