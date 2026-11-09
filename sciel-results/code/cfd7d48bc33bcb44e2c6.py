"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            calls_made += 1
            return float(func(x))

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_remaining):
            nonlocal calls_made, global_best_x, global_best_f

            n = dim
            # Population size
            m = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = m // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()

            # Step size adaptation
            cs = 1 / (n + 4)
            ds = 1 + 2 * max(0, mueff / (n + 2) - 1) + cs
            # Covariance matrix adaptation
            c1 = (2 + 2 * mueff / (n + 1) ** 2) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1) + cs)

            # State
            xmean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            Csqrt = np.eye(n)
            Cinv = np.eye(n)
            Csqrtinv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial evaluation
            f_mean = safe_func(xmean)
            if f_mean < global_best_f:
                global_best_f = f_mean
                global_best_x = xmean.copy()

            for gen in range(budget_remaining // m + 1):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                f_offspring = []
                for i in range(m):
                    if calls_made >= budget:
                        break
                    z = np.random.standard_normal(n)
                    # Use Cholesky-like update; for simplicity use Csqrt @ z
                    # We'll approximate Csqrt using eigen decomposition periodically
                    y = Csqrt @ z
                    x_new = xmean + sigma * y
                    # Clip to box
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_func(x_new)
                    offspring.append(x_new)
                    f_offspring.append(f_new)
                    if f_new < global_best_f:
                        global_best_f = f_new
                        global_best_x = x_new.copy()

                if len(offspring) < mu:
                    break

                # Sort offspring by fitness
                idx = np.argsort(f_offspring)
                offspring = [offspring[i] for i in idx]
                f_offspring = [f_offspring[i] for i in idx]

                # Update mean
                xmean_new = sum(w[i] * offspring[i] for i in range(mu))

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma * chi_n) * (Csqrtinv @ (xmean_new - xmean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // m + 1) + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * (xmean_new - xmean) / sigma
                delta_sig = (1 + (1 - c1) * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)) * (1 - c1)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1) * np.eye(n)) + cmu * sum(w[i] * np.outer(offspring[i] - xmean, offspring[i] - xmean) for i in range(mu))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Eigen decomposition to get Csqrt and Cinv
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    Cinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                    Csqrtinv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    # Fallback: reset C
                    C = np.eye(n)
                    Csqrt = np.eye(n)
                    Cinv = np.eye(n)
                    Csqrtinv = np.eye(n)

                xmean = xmean_new
                f_mean = f_offspring[0]

            return xmean, f_mean

        # Multi-start strategy: use several different starting points
        rng = np.random.default_rng(123)
        n_starts = 3
        per_start_budget = budget // n_starts

        for s in range(n_starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Different starting points
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Initial sigma
            sigma0 = 1.5 if s == 0 else 2.0

            remaining = budget - calls_made
            if remaining <= 0:
                break

            cma_es(x0, sigma0, remaining)

        # Final return
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = safe_func(global_best_x)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
