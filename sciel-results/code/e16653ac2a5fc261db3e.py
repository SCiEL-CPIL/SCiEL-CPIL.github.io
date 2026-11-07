"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with adaptive restarts for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Precompute CMA-ES parameters
        # Standard CMA-ES parameters
        def cma_es(func, x0, sigma0, max_evals, rng):
            dim = len(x0)
            # Population size
            lam = int(4 + 3 * np.log(dim))
            # Learning rates
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
            cmu = min(1 - cc, 2 * (dim + 1.5 / dim - 2) / ((dim + 2) ** 2 + 6))
            # Step size adaptation
            cs = (dim + 1.7 - 1) / (dim + 5.5 + cs)
            ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
            # Standard deviation update
            hsig = (1 + 2 / (dim + 1)) ** (cs / 2)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Covariance matrix
            C = np.eye(dim)
            Cinv = np.eye(dim)
            # Eigen decomposition for sqrt(C) and inv(sqrt(C))
            evals, evecs = np.linalg.eigh(C)
            sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            invsqrtC = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T

            # Mean
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            ps = np.zeros(dim)
            ph = np.zeros(dim)
            count_evals = 1  # x0 already evaluated

            best_x = mean.copy()
            best_f = func(mean)

            for gen in range(max_evals // lam + 1):
                if count_evals >= max_evals:
                    break

                # Generate offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    # Sample from N(0, C)
                    z = rng.standard_normal(dim)
                    x_new = mean + sigma * (sqrtC @ z)
                    # Clip to bounds
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    count_evals += 1
                    offspring.append(x_new)
                    offspring_f.append(f_new)

                # Sort by fitness
                sorted_idx = np.argsort(offspring_f)
                mu = lam // 2  # Number of parents
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mueff = 1 / np.sum(w ** 2)

                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * offspring[sorted_idx[i]]

                # CMA update
                dcorr = 1 + 2 * max(0, np.sqrt((mueff + 1) / (dim + 1) - 1) - 1) + cc
                C += (1 / (mueff * (1 - cc * (2 - cc) * dcorr))) * np.outer(y_w, y_w) * (1 + (1 - cc) * cc * dcorr)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update eigen decomposition
                evals, evecs = np.linalg.eigh(C)
                # Clip small eigenvalues
                evals = np.clip(evals, 1e-16, None)
                sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                invsqrtC = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T

                # Update step size
                y_sigma = invsqrtC @ y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma / sigma
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count_evals // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                if hsig:
                    ph = (1 - ds) * ph + np.sqrt(ds * (2 - ds) * mueff) * y_sigma / sigma
                C = (1 + mu * (cmu - mu / 2) * (1 - cc) + (0 if hsig else 0)) * C + (1 - (1 - cmu) * (1 - cc) ** 2) * np.outer(ph, ph)
                # Simplified: just keep C updated as above, skip ph update for stability
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                # Update mean
                mean = mean + cc * y_w
                mean = np.clip(mean, -5, 5)

                # Track best
                for i in range(lam):
                    if offspring_f[i] < best_f:
                        best_f = offspring_f[i]
                        best_x = offspring[i].copy()

                # Restart condition: stagnation
                if gen > 10 and np.all(np.abs(evals) < 1e-10):
                    break

            return best_x, best_f, count_evals

        # Multi-start strategy
        best_x = None
        best_f = np.inf
        used_evals = 0

        # Initial point
        x0 = np.zeros(dim)
        f0 = func(x0)
        used_evals += 1
        best_x = x0.copy()
        best_f = f0

        # Try a few different starting points with CMA-ES
        start_points = [
            np.zeros(dim),
            np.ones(dim) * 2,
            np.ones(dim) * -2,
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
        ]

        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            if not any(np.allclose(sp, us, atol=0.1) for us in unique_starts):
                unique_starts.append(sp)

        remaining_budget = budget - used_evals
        n_starts = min(len(unique_starts), max(1, remaining_budget // 100))

        for i in range(n_starts):
            if remaining_budget <= 0:
                break
            sp = unique_starts[i]
            # If we already evaluated this point, use that value
            if i == 0:
                f_sp = f0
                evals_for_this = 0
            else:
                f_sp = func(sp)
                remaining_budget -= 1
                evals_for_this = 1
                if f_sp < best_f:
                    best_f = f_sp
                    best_x = sp.copy()

            alloc = min(remaining_budget - evals_for_this, max(50, remaining_budget // (n_starts - i)))
            if alloc < 10:
                break

            x_best, f_best, evals_used = cma_es(func, sp, 1.0, alloc, rng)
            remaining_budget -= evals_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
