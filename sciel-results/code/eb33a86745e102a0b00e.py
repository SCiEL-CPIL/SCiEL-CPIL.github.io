"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Track global best
        best_x = None
        best_f = float('inf')

        # Generate diverse starting points
        start_points = []
        # Origin
        start_points.append(np.zeros(dim))
        # Random points
        rng = np.random.default_rng(42)
        for _ in range(7):
            start_points.append(rng.uniform(-4, 4, dim))
        # Some shifted points
        start_points.append(np.ones(dim) * 2.0)
        start_points.append(-np.ones(dim) * 2.0)
        start_points.append(rng.uniform(-3, 3, dim))

        # Shuffle start points
        perm = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in perm]

        # Budget per restart
        num_restarts = len(start_points)
        budget_per_restart = max(20, budget // num_restarts)

        for idx, x0 in enumerate(start_points):
            if time.time() > deadline - 5:
                break

            # CMA-ES parameters
            x = x0.copy()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # Covariance matrix (diagonal initially)
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (1 + 3 / (dim + 1)) / (dim + 3)
            cmu = 2 / ((dim + 1)**2 + 2)
            cc = (4 + dim / dim) / (4 + dim + dim / dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # Success count for sigma adaptation
            success_count = 0
            total_count = 0
            window = 5

            # Population size
            lambda_pop = 4 + int(3 * np.log(dim))
            lambda_pop = min(lambda_pop, 12)

            remaining = budget_per_restart
            # Track time for this restart
            restart_deadline = min(deadline - 3, time.time() + 60)

            for gen in range(remaining):
                if time.time() > restart_deadline:
                    break

                # Generate offspring
                offspring = []
                for k in range(lambda_pop):
                    # Sample from multivariate normal
                    z = rng.standard_normal(dim)
                    # Transform using Cholesky of C
                    L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append((y, z))

                # Evaluate offspring
                for y, z in offspring:
                    fy = func(y)
                    offspring[offspring.index((y, z))] = (y, fy, z)

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])

                # Update evolution paths
                mu_eff = sum(1 / (i + 1) for i in range(min(2 * lambda_pop // 3, lambda_pop)))
                weights = [1 / mu_eff / (i + 1) for i in range(min(2 * lambda_pop // 3, lambda_pop))]

                # Weighted mean of top offspring
                x_new = np.zeros(dim)
                for i, (y, fy, z) in enumerate(offspring[:min(2 * lambda_pop // 3, lambda_pop)]):
                    x_new += weights[i] * y

                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * (
                    (C ** -0.5) @ (x_new - x) / sigma
                )

                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) <
                        (1.4 + 2 / (dim + 1)) * chi_n)
                if hsig:
                    pc = (1 - cc) * pc + cc * (x_new - x) / sigma

                # Update C
                if hsig:
                    rank_pc = np.outer(pc, pc)
                else:
                    rank_pc = None

                rank_mu = np.zeros((dim, dim))
                for i, (y, fy, z) in enumerate(offspring[:min(2 * lambda_pop // 3, lambda_pop)]):
                    diff = (y - x) / sigma
                    rank_mu += weights[i] * np.outer(diff, diff)

                C = (1 + cmu * mu_eff - 1) * C + cmu * rank_mu
                if rank_pc is not None:
                    C += cmu * (1 / cmu - 1 + cc) * rank_pc

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Accept if best offspring is better
                best_offspring = offspring[0]
                if best_offspring[1] < fx:
                    x, fx = best_offspring[0], best_offspring[1]
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                total_count += 1

                # Adaptive sigma using 1/5 rule
                if total_count >= window:
                    rate = success_count / total_count
                    if rate > 0.5:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 0.95
                    success_count = 0
                    total_count = 0

                sigma = max(1e-8, min(sigma, 10.0))

                # Check if converged
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
