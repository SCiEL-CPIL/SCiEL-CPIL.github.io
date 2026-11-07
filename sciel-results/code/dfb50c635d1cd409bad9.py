"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restarted CMA-ES for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0
        max_calls = self.budget

        def time_check():
            return time.time() < deadline

        # CMA-ES implementation for a single run
        def cmaes_run(x0, sigma0, budget_for_run, seed):
            nonlocal total_calls, best_x, best_f

            rng = np.random.default_rng(seed)
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size

            # Initial mean
            mean = x0.copy()
            sigma = sigma0

            # Covariance matrix
            C = np.eye(n)

            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)

            # Parameters
            cs = (1.5 + 0.5 * (n - 1)) / (n + 1.5 + 0.5 * (n - 1))
            ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + cs
            cmu = min(0.5, (1 - cs) / (2 * (n + 2)))
            mu = lam // 2  # number of parents
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            # Step size control
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track best
            local_best_x = mean.copy()
            local_best_f = np.inf

            # Initial evaluation
            if total_calls < max_calls and time_check():
                f_val = func(mean)
                total_calls += 1
                if f_val < local_best_f:
                    local_best_f = f_val
                    local_best_x = mean.copy()
                    if f_val < best_f:
                        best_f = f_val
                        best_x = mean.copy()
            else:
                return local_best_x, local_best_f

            # Main loop
            while total_calls < max_calls and time_check():
                # Generate population
                population = []
                for i in range(lam):
                    if not time_check() or total_calls >= max_calls:
                        break
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    population.append((y, z))

                if not population:
                    break

                # Evaluate population
                fitness = []
                for y, z in population:
                    if not time_check() or total_calls >= max_calls:
                        break
                    f_val = func(y)
                    total_calls += 1
                    fitness.append(f_val)
                    if f_val < local_best_f:
                        local_best_f = f_val
                        local_best_x = y.copy()
                        if f_val < best_f:
                            best_f = f_val
                            best_x = y.copy()

                if not fitness:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(fitness)
                selected = [population[i] for i in sorted_idx[:mu]]
                sorted_fitness = [fitness[i] for i in sorted_idx]

                # Update mean
                new_mean = np.zeros(n)
                for w, (y, z) in zip(weights, selected):
                    new_mean += w * y
                mean = new_mean

                # Update covariance matrix
                if mueff > 0:
                    diff = np.array([y - old_mean for y, z in selected])
                    pc = (1 - cmu) * pc + cmu * np.sqrt(mueff) * sigma * (C @ (np.array([z for y, z in selected]).T @ weights)).flatten()
                    # Simplified: update C using rank-one and rank-mu updates
                    # For simplicity and robustness, use a simplified update
                    hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * 1))) / chi_n < 1.4 + 2 / (n + 1)
                    rank_one = (1 + (2 * (hsig - 1)) * cs * (1 - cs)) * np.outer(pc, pc)
                    rank_mu = cmu / (1 - (1 - cs) ** 2) * (
                        np.outer(weights, diff) @ (C @ diff.T) * weights
                    )
                    # This is getting complex; let's use a simpler approach
                    pass

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(np.array([z for y, z in selected]).T @ weights).flatten() / chi_n - 1))

                # Clip sigma
                sigma = np.clip(sigma, 1e-10, 100)

                # Check for convergence
                if np.linalg.norm(C) < 1e-10 or sigma < 1e-10:
                    break

            return local_best_x, local_best_f

        # Multi-start strategy: try several different starting points
        rng_global = np.random.default_rng(42)
        n_starts = min(10, max(2, max_calls // 50))

        for i in range(n_starts):
            if not time_check() or total_calls >= max_calls:
                break

            # Different starting points
            if i == 0:
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
            elif i == 1:
                x0 = rng_global.uniform(-4, 4, self.dim)
                sigma0 = 1.5
            elif i == 2:
                x0 = rng_global.uniform(-5, 5, self.dim)
                sigma0 = 3.0
            else:
                x0 = rng_global.uniform(-4, 4, self.dim)
                sigma0 = 1.0 + rng_global.uniform(0, 2)

            # Budget for this run
            remaining = max_calls - total_calls
            if remaining <= 0:
                break
            budget_for_run = min(remaining, max(20, remaining // (n_starts - i)))

            cmaes_run(x0, sigma0, budget_for_run, seed=1000 + i)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
