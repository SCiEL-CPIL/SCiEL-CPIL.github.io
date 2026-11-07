"""Optimizer for BBOB-style f11 (Discus) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s

        # CMA-ES parameters
        dim = self.dim
        budget = self.budget

        # Population size (lambda)
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # number of parents

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        sigma = 2.0

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Track best
        best_x = x.copy()
        best_fx = fx

        for iteration in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            population = []
            fitnesses = []

            for i in range(lam):
                # Generate candidate
                z = np.random.randn(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append(y)
                fitnesses.append(fy)

            # Check time again
            if time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]

            # Update best
            if fitnesses[0] < best_fx:
                best_fx = fitnesses[0]
                best_x = population[0].copy()

            # Recombination
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += weights[i] * population[i]

            # Update path and covariance
            diff = new_x - x
            x = new_x
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * diff / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * B @ (diff / sigma)

            # Update covariance
            delta_hs = (np.linalg.norm(ps) ** 2 / (1 - (1 - cs) ** (2 * (iteration + 1))) >
                        (1 + 1.4 / dim) * dim)
            C = ((1 - c1 - cmu) * C +
                 c1 * (pc @ pc + (delta_hs - 1 + cc * (2 - cc)) * C) +
                 cmu * np.sum([np.outer(population[i] - x, population[i] - x) for i in range(mu)], axis=0) * weights)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update sigma
            sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(2 * dim * (1 - (1 - cs) ** (iteration + 1))) - 1))

            # Check convergence
            if np.all(np.abs(population[0] - population[-1]) < 1e-10):
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
