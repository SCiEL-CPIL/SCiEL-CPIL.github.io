"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Büche-Rastrigin (f4) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit

        # We'll use CMA-ES with adaptive sigma and covariance matrix
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = int(4 + 3 * np.log(dim))  # for dim=5: 4+3*1.609=8.83 -> 8, use 9 for odd
        lam = max(lam, 6)

        # CMA-ES parameters
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Chi-normal for sigma adaptation
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Restart mechanism: run multiple CMA-ES instances with different starting points
        # and adapt sigma based on progress
        n_restarts = 5
        budget_per_restart = max(1, (budget - n_restarts) // n_restarts)

        for restart in range(n_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                # Start near origin for first restart (Büche-Rastrigin has global min near origin)
                x = np.zeros(dim)
                sigma = 2.0
            else:
                # Use best point found so far, or random
                if best_x is not None and restart < 3:
                    x = best_x.copy()
                    sigma = 1.0
                else:
                    rng = np.random.default_rng(restart * 1000)
                    x = rng.uniform(-5, 5, dim)
                    sigma = 2.0

            # Evaluate initial point
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES main loop
            C = np.eye(dim)  # covariance matrix
            pc = np.zeros(dim)  # evolution path for C
            ps = np.zeros(dim)  # evolution path for sigma

            # Initialize population
            rng = np.random.default_rng(restart * 1000 + 42)
            pop = np.array([np.clip(x + sigma * rng.multivariate_normal(np.zeros(dim), C), -5, 5)
                           for _ in range(lam)])
            fvals = np.array([func(xi) for xi in pop])
            calls_used += lam

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Mean update
            mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update paths
            y = (mean - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y

            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) - delta_hsig) * C \
                + c_1 * (np.outer(pc, pc) - C) \
                + c_mu * np.sum(weights[:mu] * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0) / sigma ** 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = mean

            # Check convergence
            if np.all(np.abs(pop - x) < 1e-10 * (1 + np.abs(x))):
                break

            # Generate next population
            while calls_used < budget and time.time() < deadline:
                # Check for stagnation: if no improvement in last lam evaluations, restart
                if fvals[0] >= best_f:
                    # Try to escape: perturb sigma
                    sigma *= 1.2
                    if sigma > 5:
                        break

                pop = np.array([np.clip(x + sigma * rng.multivariate_normal(np.zeros(dim), C), -5, 5)
                               for _ in range(lam)])
                fvals = np.array([func(xi) for xi in pop])
                calls_used += lam

                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()

                mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                y = (mean - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_c * (2 - c_c) - delta_hsig) * C \
                    + c_1 * (np.outer(pc, pc) - C) \
                    + c_mu * np.sum(weights[:mu] * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0) / sigma ** 2

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                x = mean

                if np.all(np.abs(pop - x) < 1e-10 * (1 + np.abs(x))):
                    break

                if time.time() > deadline - 10:
                    break

        # Final polish: local search around best point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            # Simple gradient-free local refinement
            step = 0.01
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        trial = best_x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls_used += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
