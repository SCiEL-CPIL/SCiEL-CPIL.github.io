"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Use multiple restarts with CMA-ES
        max_restarts = 10
        calls_per_restart = max(50, budget // max_restarts)

        rng = np.random.default_rng(12345)

        for restart in range(max_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize: center at best found so far, or random for first restart
            if restart == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Start near the best point found so far with some perturbation
                x0 = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)

            fx = func(x0)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

            # CMA-ES parameters
            lambda_ = max(4, dim)  # population size
            sigma = 2.0 / dim * 5.0  # initial step size
            c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
            cmu = min(0.5, (lambda_ // 2 - 1 + 2 / (dim + lambda_)) / ((dim + 2) ** 2 + lambda_))
            cs = (lambda_ + 2) / (dim + lambda_ + 5)
            ds = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + cs
            cc = (4 + dim / lambda_) / (dim + 4 + 2 * lambda_ / dim)
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Covariance matrix and its inverse
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)

            # Evolution path
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Mean
            mean = x0.copy()
            f_mean = fx

            # Budget for this restart
            remaining = budget - total_calls
            max_iters = min(remaining, 200)

            for iteration in range(max_iters):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                offspring = np.zeros((lambda_, dim))
                fitness = np.zeros(lambda_)

                for i in range(lambda_):
                    # Sample from multivariate normal using C_sqrt
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    offspring[i] = x
                    fitness[i] = func(x)
                    total_calls += 1

                    if fitness[i] < best_f:
                        best_f = fitness[i]
                        best_x = x.copy()

                # Sort by fitness
                idx = np.argsort(fitness)
                fitness_sorted = fitness[idx]
                offspring_sorted = offspring[idx]

                # Update mean (weighted)
                weights = np.zeros(lambda_)
                mu_eff = lambda_ // 2
                top_mu = lambda_ // 2
                for i in range(top_mu):
                    weights[i] = np.log(top_mu + 0.5) - np.log(i + 1)
                weights /= weights.sum()

                new_mean = weights @ offspring_sorted
                new_mean = np.clip(new_mean, -5, 5)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    C_inv @ (sigma * (new_mean - mean))
                )
                norm_ps = np.linalg.norm(ps)

                hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) < \
                       (1.4 + 2 / (dim + 1)) * chi

                if hsig:
                    pc = (1 - cc) * pc + cc / sigma * (new_mean - mean)

                # Update covariance matrix
                delta = (2 * hsig - 1) * cc * (2 - cc)
                C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(ps, ps) + \
                    cmu / sigma ** 2 * sum(
                        weights[i] * np.outer(
                            (offspring_sorted[i] - mean) / sigma,
                            (offspring_sorted[i] - mean) / sigma
                        ) for i in range(top_mu)
                    )

                # Ensure positive semi-definite and compute square root
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_sqrt = eigvecs @ np.sqrt(np.diag(eigvals)) @ eigvecs.T
                    C_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv = np.eye(dim)

                # Update sigma
                sigma *= np.exp(cs / ds * (norm_ps / chi - 1))

                # Update mean
                mean = new_mean

                # Early stopping if sigma is too small or no improvement
                if sigma < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
