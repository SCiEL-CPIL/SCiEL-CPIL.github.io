"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of recombination weights

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs

        # Covariance matrix adaptation
        cmc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cm1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cm2 = min(1 - cm1, 2 * ((mueff + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + mueff))

        # Initial values
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Initial step size, covering the box reasonably
        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial evaluation
        calls = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls += 1

        if calls >= budget:
            return best_x, best_f

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C).T
            W = np.zeros((lambda_, dim))
            F = np.zeros((lambda_, dim))
            for i in range(lambda_):
                zi = np.random.standard_normal(dim)
                wi = sigma * (B @ zi)
                W[i] = zi
                F[i] = x + wi

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    fvals[i] = fvals[i - 1] if i > 0 else best_f
                    continue
                fvals[i] = func(F[i])
                calls += 1
                if calls >= budget:
                    break
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = F[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            W = W[idx]
            F = F[idx]
            fvals = fvals[idx]

            # Update mean
            xold = x.copy()
            x_new = np.sum(weights[:, None] * F[:mu], axis=0)
            x = x_new

            # Update evolution paths
            ydiff = (x - xold) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ ydiff)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls))) < (1.4 + 2 / (dim + 1)) * chiN)
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * ydiff

            # Update covariance matrix
            delta_d = 1 + ds * (cs - 1) + (2 * (1 - hsig) / cs - 1) * cmc
            C = (1 + cmc * (delta_d - 1) - cm1 - cm2) * C \
                + cmc * (delta_d - 1) * np.outer(pc, pc) \
                + cm1 * np.sum(weights[:, None, None] * np.outer(F[:mu] - xold, F[:mu] - xold), axis=0) \
                + cm2 * np.outer(pc, pc)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(dim)

            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))

            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Project back to box if needed (CMA-ES usually handles this via boundary reflection or clipping)
            # For simplicity, we clip the mean and generated points. The evaluator clips func inputs anyway.
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
