"""CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 20 and adaptive stopping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size 20 allows for 24 generations in 500 budget (1 + 24*20 = 481)
        # This provides a good balance between exploration and exploitation.
        lambda_ = 20
        mu = lambda_ // 2

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        dsigma = cs + 2 / (dim + 1)

        # Initial sigma: 2.5 is slightly larger than standard 1.0 or 2.0 to ensure
        # the initial population covers the [-5, 5]^5 box more effectively.
        sigma = 2.5

        # Initial mean
        xmean = np.zeros(dim)
        fx_mean = func(xmean)

        # Best solution tracking
        best_x = xmean.copy()
        best_fx = fx_mean

        # PCA matrices
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv = np.eye(dim)
        Cinv_sqrt = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Max generations
        max_gens = (budget - 1) // lambda_

        for gen in range(max_gens):
            # Sample population
            pop = np.zeros((lambda_, dim))
            pop_fx = np.zeros(lambda_)

            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = xmean + sigma * (Csqrt @ z)
                pop[i] = np.clip(pop[i], -5, 5)
                pop_fx[i] = func(pop[i])

            # Sort
            order = np.argsort(pop_fx)
            pop_sorted = pop[order]
            pop_fx_sorted = pop_fx[order]

            # Update best
            if pop_fx_sorted[0] < best_fx:
                best_fx = pop_fx_sorted[0]
                best_x = pop_sorted[0].copy()

            # Recombination
            xnew = np.dot(weights, pop_sorted[:mu])

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ ((xnew - xmean) / sigma))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / np.sqrt(mueff)
            if hsig < 1.4 + 2 / (dim + 1):
                pc = (1 - cc) * pc + cc / (1 - cc) * (xnew - xmean) / sigma
            else:
                pc = (1 - cc) * pc

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            # Use the rank-mu update
            outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                outer_sum += np.outer(pop_sorted[i] - xmean, pop_sorted[i] - xmean)
            
            C = (1 + c1 * delta_hsig * np.outer(ps, ps) - c1) * C + \
                (cmu / (sigma ** 2)) * outer_sum

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Eigen-decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                Cinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                Csqrt = np.eye(dim)
                Cinv_sqrt = np.eye(dim)
                Cinv = np.eye(dim)

            # Update mean
            xmean = xnew

            # Adapt sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))

            # Stopping criteria
            if sigma < 1e-12:
                break
            if np.max(np.abs(xmean)) > 1e6:
                break

        return best_x, best_fx


# EVOLVE-BLOCK-END
