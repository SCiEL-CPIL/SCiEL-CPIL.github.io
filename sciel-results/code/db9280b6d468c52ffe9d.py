"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and time guard for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_f = np.inf

        # We'll do multiple restarts of CMA-ES with different starting points
        # and decreasing budgets per restart

        # Strategy:
        # - First restart: budget 300, start near center
        # - Second restart: budget 120, random start
        # - Third restart: budget 60, random start
        # - Remaining: local refinement around best found

        restarts = [
            (300, "center"),
            (120, "random"),
            (60, "random"),
            (20, "refine"),
        ]

        for r_budget, mode in restarts:
            if time.time() > deadline:
                break
            if r_budget <= 0:
                break

            # Determine starting point
            if mode == "center":
                x0 = np.zeros(dim)
                # Add small perturbation to avoid exact symmetry
                x0 = x0 + 0.1 * rng.standard_normal(dim)
            elif mode == "random":
                x0 = rng.uniform(-4, 4, dim)
            elif mode == "refine":
                if best_x is not None:
                    x0 = best_x + 0.5 * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, dim)

            # CMA-ES parameters
            # Population size: 4 + floor(3 * ln(dim))
            n = 4 + int(3 * np.log(dim))
            # Step size
            sigma = 1.5
            # Max iterations for this restart
            max_iter = r_budget // n

            # Initialize CMA-ES state
            mean = x0.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)  # We'll use a simplified approach
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (n + 2) / (n + 5)
            cmu = 2 / ((n + 4) ** 2 + 10)
            c_sigma = (n + 1.3) / (n + 5 + 2 * cmu)
            d_sigma = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            cc = (4 + n / cs) / (4 + cs)
            c1 = (2 + 2 * cmu / n) / ((n + 2) ** 2 + 2)
            cmu_eff = cmu / ((n + 2) ** 2 + 2)

            # For simplicity and speed, use a diagonal CMA-ES variant (sep-CMA)
            # which is faster in low dimensions
            diag = np.ones(dim)  # diagonal of C
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            local_best_x = x0.copy()
            local_best_f = func(x0)

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            calls_made = 1

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= r_budget:
                    break

                # Generate population
                # In diagonal CMA, we sample from N(mean, sigma^2 * diag)
                # But let's use a simpler approach: sample directions, scale by sigma * sqrt(diag)
                W = rng.standard_normal((n, dim))
                # Apply diagonal scaling
                W = W * np.sqrt(diag)
                # Rank-mu update would be complex; use simple (1+lambda) or (mu+lambda)
                # For speed, use (mu + lambda) with mu = n//2

                # Evaluate population
                pop = mean[None, :] + sigma * W
                pop = np.clip(pop, -5, 5)

                # Evaluate all
                fvals = np.empty(n)
                for i in range(n):
                    if time.time() > deadline:
                        break
                    if calls_made >= r_budget:
                        break
                    fvals[i] = func(pop[i])
                    calls_made += 1
                    if fvals[i] < local_best_f:
                        local_best_f = fvals[i]
                        local_best_x = pop[i].copy()
                        if local_best_f < best_f:
                            best_f = local_best_f
                            best_x = pop[i].copy()

                if calls_made >= r_budget:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                # Use mu = n//2 best solutions
                mu = max(1, n // 2)
                mu_idx = order[:mu]

                # Update weights (equal weights for simplicity)
                w = np.ones(mu) / mu
                mu_eff = 1.0 / np.sum(w ** 2)

                # New mean
                new_mean = np.sum(w[:, None] * pop[mu_idx], axis=0)

                # Update step size (CSA)
                # ps = (1 - cs) * ps + cs / chi_n * sqrt(mu_eff) * C^{-1/2} (new_mean - mean) / sigma
                # For diagonal: C^{-1/2} = 1/sqrt(diag)
                ps = (1 - cs) * ps + cs / chi_n * np.sqrt(mu_eff) * (new_mean - mean) / (sigma * np.sqrt(diag))

                # hsig: check if ps is within expected range
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update covariance (diagonal only)
                # pc = (1 - cc) * pc + cc * hsig * (new_mean - mean) / sigma
                if hsig:
                    pc = (1 - cc) * pc + cc * (new_mean - mean) / sigma

                # For diagonal CMA, update diag based on weighted second moments
                # y_i = (pop[i] - mean) / sigma
                y = (pop[mu_idx] - mean[None, :]) / sigma
                # diag_new = (1 - c1 - cmu_eff) * diag + c1 * (pc * pc + (cc*(2-cc))*diag) + cmu_eff * sum(w_i * y_i^2)
                diag_new = (1 - c1 - cmu_eff) * diag
                diag_new += c1 * (pc ** 2 + cc * (2 - cc) * diag)
                diag_new += cmu_eff * np.sum(w[:, None] * y ** 2, axis=0)
                diag = np.clip(diag_new, 1e-10, None)

                mean = new_mean

            # After restart, if we have time, do a few local steps around best
            # This is handled by the next restart if mode is "refine"

        # Final: ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
