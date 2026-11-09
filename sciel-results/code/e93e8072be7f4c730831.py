"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, adaptive step size, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 - 1 / dim + 2 * mueff / dim) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        def cmaes(x0, sigma0, remaining_budget, rng, func):
            n = 0
            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            best_x = x0.copy()
            best_f = func(x0)
            n += 1
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            while n < remaining_budget and time.time() < deadline:
                # Generate population
                pop = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    pop[i] = xmean + sigma * C @ rng.standard_normal(dim)
                    pop[i] = np.clip(pop[i], -5, 5)

                # Evaluate
                fvals = np.array([func(x) for x in pop])
                n += lambda_

                # Update best
                idx = np.argmin(fvals)
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = pop[idx].copy()

                # Sort
                order = np.argsort(fvals)
                y_sorted = pop[order]
                f_sorted = fvals[order]

                # Weighted mean
                xnew = np.dot(weights, y_sorted[:mu])

                # Update step size control
                xnorm = xnew - xmean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * xnorm
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n + 1) / (dim * cs))) <
                        (1.4 + 2 / (dim + 1)) * chi_n) if np.linalg.norm(ps) > 0 else True

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * xnorm

                # Update covariance matrix
                delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) * (1 - cc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(y_sorted[i] - xmean, y_sorted[i] - xmean)
                rank_one = np.outer(pc, pc)
                C = (1 + c1 * (rank_one - C) + cmu * rank_mu) * (1 - c1 * delta_d * (np.trace(rank_one) / dim - 1))

                # Ensure C is positive definite and symmetric
                C = (C + C.T) / 2
                # Regularize
                C += 1e-12 * np.eye(dim)

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 100)

                xmean = xnew

            return best_x, best_f, n

        # Multiple restarts with different initializations
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        # First try: CMA-ES from center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        bx, bf, used = cmaes(x0, sigma0, budget, rng, func)
        calls_made += used
        if bf < best_f:
            best_f = bf
            best_x = bx

        # If we have budget left, try another restart from a random point
        if calls_made < budget and time.time() < deadline:
            x0 = rng.uniform(-4, 4, dim)
            sigma0 = 1.5
            bx, bf, used = cmaes(x0, sigma0, budget - calls_made, rng, func)
            calls_made += used
            if bf < best_f:
                best_f = bf
                best_x = bx

        # If still have budget, try a third restart
        if calls_made < budget and time.time() < deadline:
            x0 = rng.uniform(-3, 3, dim)
            sigma0 = 1.0
            bx, bf, used = cmaes(x0, sigma0, budget - calls_made, rng, func)
            calls_made += used
            if bf < best_f:
                best_f = bf
                best_x = bx

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
