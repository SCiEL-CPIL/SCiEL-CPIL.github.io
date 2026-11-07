"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step and covariance for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0=1.0, max_iter=200):
            nonlocal calls_made, best_f, best_x

            # CMA-ES parameters
            N = dim  # population size
            mu = N // 2  # number of parents
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialization
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)  # inverse square root of C
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Precompute
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for iteration in range(max_iter):
                if calls_made >= budget - N or time.time() > deadline:
                    return

                # Generate population
                population = []
                for i in range(N):
                    z = rng.standard_normal(dim)
                    y = C_sqrt_inv @ z
                    x_i = mean + sigma * y
                    x_i = np.clip(x_i, -5.0, 5.0)
                    population.append(x_i)

                # Evaluate population
                f_vals = [eval_func(x) for x in population]

                # Sort by fitness
                sorted_idx = np.argsort(f_vals)
                population = [population[i] for i in sorted_idx]
                f_vals = [f_vals[i] for i in sorted_idx]

                # Update mean with weighted recombination
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += w[i] * population[i]
                mean = new_mean

                # Update covariance matrix
                diff = np.zeros((dim, mu))
                for i in range(mu):
                    diff[:, i] = (population[i] - mean) / sigma

                # Rank-1 update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (new_mean - mean) / sigma)
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1 + 2 / (dim + 1)) * chi_n else 0

                # Rank-mu update
                delta_h = (1 - c_c) ** (2 * (iteration + 1))
                C = (1 + c_1 * h_sigma * (2 - c_sigma) + c_mu) * C
                C -= c_1 * (1 + h_sigma * (2 - c_sigma) - c_c) * np.outer(p_c, p_c)
                C -= c_mu * np.outer(p_c, p_c) * 0  # simplified

                # Update C with rank-mu
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_sigma, p_sigma) / mu_eff * (2 - c_sigma))
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(diff[:, i], diff[:, i])

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update path evolution
                p_c = (1 - c_c) * p_c + c_c * (new_mean - mean) / sigma

                # Update C_sqrt_inv (approximate)
                eigvals_inv, eigvecs_inv = np.linalg.eigh(C)
                C_sqrt_inv = eigvecs_inv @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_inv, 1e-12))) @ eigvecs_inv.T

                # Convergence check
                if np.max(f_vals[:mu]) - np.min(f_vals[:mu]) < 1e-10:
                    break

        # Initial exploration: evaluate several diverse points
        initial_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for pt in initial_points:
            if calls_made >= budget:
                break
            eval_func(pt)

        # Run CMA-ES from the best initial point
        if best_x is not None and calls_made < budget - 20 and time.time() < deadline:
            cma_es(best_x.copy(), sigma0=1.5, max_iter=150)

        # If we have budget left, try another CMA-ES from a different region
        if calls_made < budget - 20 and time.time() < deadline:
            alt_start = rng.uniform(-4, 4, dim)
            cma_es(alt_start, sigma0=2.0, max_iter=80)

        # Final refinement with smaller step
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline:
            cma_es(best_x.copy(), sigma0=0.5, max_iter=50)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
