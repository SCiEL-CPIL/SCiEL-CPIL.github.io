"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es_run(x0, sigma0, max_evals, pop_size=10):
            """Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations."""
            nonlocal calls_made, best_f, best_x, check_time

            lam = pop_size  # population size
            mu = max(1, lam // 2)  # number of parents
            # Weights (logarithmic)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

            # CMA-ES parameters
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))

            evals_in_run = 0
            last_best = np.inf
            stagnation_counter = 0
            stagnation_limit = 20  # generations without improvement

            while evals_in_run < max_evals and calls_made < budget and check_time():
                # Generate population
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity
                    L = np.eye(dim)

                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    if calls_made >= budget or not check_time():
                        break
                    fy = evaluate(y)
                    evals_in_run += 1
                    offspring_f.append(fy)

                if len(offspring_f) < lam:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                y_sorted = [offspring[i] for i in idx]
                f_sorted = [offspring_f[i] for i in idx]

                # Check for stagnation
                if f_sorted[0] < last_best - 1e-6:
                    last_best = f_sorted[0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # If stagnating, restart from best known point with larger sigma
                if stagnation_counter > stagnation_limit:
                    if best_x is not None:
                        x_mean = best_x.copy()
                        sigma = max(sigma * 1.5, 1.0)
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        stagnation_counter = 0
                    else:
                        break

                # Update mean
                x_new_mean = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0)

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ (x_new_mean - x_mean))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 0.01, 10.0)

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma

                delta_heavy = (1 - c_1) * (1 - c_mu) + c_1 * c_mu
                C = ((1 + c_1 * (c_mu - 1) - c_mu) * C
                     + c_1 / c_1 * (np.outer(p_c, p_c) - C)
                     + c_mu / c_1 * np.sum([
                         weights[i] * np.outer(y_sorted[i] - x_mean, y_sorted[i] - x_mean)
                         for i in range(mu)
                     ], axis=0))

                # Ensure symmetry
                C = (C + C.T) / 2

                x_mean = x_new_mean

            return x_mean, sigma

        # Phase 1: Evaluate diverse starting points
        n_starts = 10
        for i in range(n_starts):
            if calls_made >= budget or not check_time():
                break
            if i < 5:
                x = rng.uniform(-4, 4, dim)
            elif i < 8:
                x = rng.uniform(-4, 4, dim)
                for d in range(dim):
                    if rng.random() < 0.4:
                        x[d] = round(x[d])
            else:
                x = rng.uniform(-3, 3, dim)
            evaluate(x)

        # Phase 2: Run CMA-ES from best point
        remaining = budget - calls_made
        if remaining > 0 and check_time() and best_x is not None:
            x0 = best_x.copy()
            sigma0 = 1.5
            # Run CMA-ES with remaining budget
            cma_es_run(x0, sigma0, remaining, pop_size=10)

        # Phase 3: If we still have budget, do a few more random restarts + short CMA
        remaining = budget - calls_made
        if remaining > 20 and check_time():
            n_extra = min(5, remaining // 20)
            for _ in range(n_extra):
                if calls_made >= budget or not check_time():
                    break
                x_new = rng.uniform(-4, 4, dim)
                evaluate(x_new)
                if x_new is not None and best_x is not None:
                    # Short CMA from new point if it improved
                    if remaining > 15:
                        cma_es_run(x_new, 1.0, min(15, budget - calls_made), pop_size=5)
                    remaining = budget - calls_made

        return best_x, best_f
# EVOLVE-BLOCK-END
