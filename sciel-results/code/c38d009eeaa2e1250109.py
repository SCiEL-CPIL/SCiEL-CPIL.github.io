"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f13 (Sharp Ridge) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls_made = [0]

        def safe_call(x):
            if calls_made[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls_made[0] += 1
                return val
            except:
                calls_made[0] += 1
                return float('inf')

        best_x = None
        best_f = float('inf')

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # lambda = 4 + 3*ln(5) ≈ 9
        mu = lam // 2  # top mu parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))
        # Step size control
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Restart strategy
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline or calls_made[0] >= budget:
                break

            # Initialize
            if restart == 0:
                x_mean = rng.uniform(-4, 4, dim)
            else:
                if best_x is not None:
                    x_mean = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                else:
                    x_mean = rng.uniform(-4, 4, dim)

            sigma = 2.0  # initial step size
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Evaluate initial population
            pop = [x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C) for _ in range(lam)]
            pop = [np.clip(x, -5, 5) for x in pop]
            fvals = [safe_call(x) for x in pop]

            for x, f in zip(pop, fvals):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # CMA-ES iterations
            stall_count = 0
            max_stall = 50

            while calls_made[0] < budget and time.time() < deadline:
                # Sort by function value
                order = np.argsort(fvals)
                pop_sorted = [pop[i] for i in order]
                fvals_sorted = [fvals[i] for i in order]

                # Update best
                for x, f in zip(pop_sorted, fvals_sorted):
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()

                # Check for stagnation
                if fvals_sorted[0] == fvals_sorted[-1]:
                    stall_count += 1
                else:
                    if fvals_sorted[0] < best_f - 1e-10:
                        stall_count = 0
                    else:
                        stall_count += 1

                if stall_count > max_stall:
                    break

                # Select top mu parents
                parents = pop_sorted[:mu]
                # Weighted recombination
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * parents[i]

                # Update evolution path for covariance matrix
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * (C @ ((x_new_mean - x_mean) / sigma))

                # Hessian update for step size control
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * ((x_new_mean - x_mean) / sigma)

                # Update covariance matrix
                delta_hsig = ((1 + cs) * np.linalg.norm(p_sigma) ** 2 / chi_n - 1) * (1 + cs)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    y_i = (parents[i] - x_mean) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                # Update step size
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))

                x_mean = x_new_mean

                # Generate new population
                pop = [x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C) for _ in range(lam)]
                pop = [np.clip(x, -5, 5) for x in pop]
                fvals = [safe_call(x) for x in pop]

        return best_x, best_f
# EVOLVE-BLOCK-END
