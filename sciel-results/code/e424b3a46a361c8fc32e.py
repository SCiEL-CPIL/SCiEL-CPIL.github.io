"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with large initial population for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(7)

        best_x = None
        best_f = float('inf')

        # CMA-ES with larger population to maintain diversity
        lambda_pop = 12  # population size
        mu = lambda_pop // 2  # number of recombination parents

        # Initialize mean at origin with large sigma
        mean = np.zeros(dim)
        sigma = 3.0

        # Initialize CMA-ES matrices
        C = np.eye(dim)  # covariance matrix
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))

        # CMA-ES parameters
        cs = (mu + 2) / (dim + mu + 5)
        cmu = 2 / ((dim + 2) ** 2 + mu)
        cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu)
        cmu = cmu - c1

        # Damping
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Precompute weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w / w.sum()

        evals_used = 0

        # Main CMA-ES loop
        while evals_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_pop, dim))
            fit = np.zeros(lambda_pop)

            # Precompute Cholesky once per iteration
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            for i in range(lambda_pop):
                if evals_used >= budget or time.time() > deadline:
                    break

                # Standard normal sample
                z = rng.standard_normal(dim)
                # Transform to N(0, C)
                y = L @ z
                # Clip to bounds
                xi = np.clip(mean + sigma * y, -5, 5)
                pop[i] = xi
                fit[i] = func(xi)
                evals_used += 1

                if fit[i] < best_f:
                    best_f = fit[i]
                    best_x = xi.copy()

            # If we broke out early, pad the population
            if evals_used < budget:
                remaining = lambda_pop - (evals_used - (lambda_pop - 1) if evals_used >= lambda_pop else 0)
                # This is getting complex, let's just break if we didn't complete the population
                if evals_used < lambda_pop:
                    break

            # Sort by fitness
            order = np.argsort(fit)
            fit_sorted = fit[order]
            pop_sorted = pop[order]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)

            # Update covariance matrix
            diff = mean - mean_old
            # Rank-one update
            C += c1 * (diff[:, None] * diff[None, :] - C)

            # Rank-mu update
            diff_mu = pop_sorted[:mu] - mean_old
            # Weighted outer products
            rank_mu = np.zeros((dim, dim))
            for k in range(mu):
                rank_mu += weights[k] * np.outer(diff_mu[k], diff_mu[k])
            C += cmu * rank_mu

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update evolution paths
            inv_sigma = 1.0 / sigma
            ps = (1 - cs) * ps + cs * inv_sigma * (L.T @ (mean - mean_old) / sigma)

            pc = (1 - cc) * pc + cc * cs / (1 - cs) * (
                (1 + cs) * (mean - mean_old) / sigma - ps
            )

            # sigma update
            hsig = (
                np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used // lambda_pop)))
            )
            hsig = 1.0 if hsig < (1.4 + 2 / (dim + 1)) * chiN else 0.0

            # Update C with pc
            C += (c1 * (hsig - 1 + cc) + cmu * (hsig - 1)) * (
                cc / (1 - cc) * C + pc
            )

            # Sigma update
            sigma *= np.exp(
                cs / chiN * (np.linalg.norm(ps) / chiN - 1)
            )

            # Keep sigma within reasonable bounds
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 15.0:
                sigma = 15.0

            # Periodically re-orthogonalize C to prevent numerical issues
            if evals_used % (lambda_pop * 5) < lambda_pop:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
