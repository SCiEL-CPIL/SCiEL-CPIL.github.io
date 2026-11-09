# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._start_time = time.time()
        deadline = self._start_time + 280  # safety margin

        def wrapped(x):
            self._call_count += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < self._best_f:
                self._best_f = f
                self._best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        def cma_run(x0, sigma0, max_calls):
            """Run CMA-ES from x0 with sigma0, using at most max_calls evaluations."""
            dim = self.dim
            n = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = n // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c = 1.0 / (dim + 4.0)
            dsigma = 0.5
            c_sigma = (dsigma + 1.0) / (dim + dsigma + 3.0)
            c_c = (4.0 + dim / c) / (dim + 4.0 + 2.0 * dim / c)
            c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            calls_used = 0

            while calls_used < max_calls:
                if time.time() >= deadline:
                    break
                # Generate population
                z = rng.standard_normal((n, dim))
                # Normalize by Mahalanobis distance
                norms = np.sqrt(np.sum((z @ C) * z, axis=1))
                z /= norms[:, None]
                # Mean of best mu
                mean_z = np.sum(weights[:, None] * z[:mu], axis=0)
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C @ mean_z)
                hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1))) <
                      (1 + 2.0 / (dim + 1)) * chi_n) and 1.0 or 0.0
                pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_z
                # Update covariance
                delta_d = (hs - 1.0) * c_c * (2 - c_c)
                C = (1.0 + c_1 * delta_d - c_1) * C + c_1 * np.outer(pc, pc) + \
                    c_mu / mu_eff * np.sum(weights[i] * np.outer(z[i], z[i]) for i in range(mu))
                # Ensure C is symmetric and positive definite
                C = 0.5 * (C + C.T)
                # Cholesky for stability
                try:
                    L = np.linalg.cholesky(C)
                    C = L @ L.T
                except np.linalg.LinAlgError:
                    # Add jitter
                    C += 1e-8 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                        C = L @ L.T
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)

                # Generate and evaluate population
                pop = x[None, :] + sigma * (z @ L.T)
                # Clip to box
                pop = np.clip(pop, -5.0, 5.0)
                # Evaluate
                vals = np.array([wrapped(pop[i]) for i in range(n)])
                calls_used += n

                # Sort
                idx = np.argsort(vals)
                x = pop[idx[0]]
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(pop[idx[0]] - x) / sigma - 1.0))
                sigma = max(1e-10, min(sigma, 50.0))

            return x, self._best_f

        # Multiple restarts with different initializations
        n_restarts = 3
        calls_per_restart = (self.budget - 1) // n_restarts
        remaining = self.budget - 1

        for r in range(n_restarts):
            if time.time() >= deadline or remaining <= 0:
                break
            # Different starting points
            if r == 0:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 3.0
            elif r == 1:
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, self.dim)
                sigma0 = 1.5

            max_calls_this = min(calls_per_restart, remaining)
            x, f = cma_run(x0, sigma0, max_calls_this)
            remaining -= max_calls_this

        # Use remaining budget for local refinement around best point
        if remaining > 0 and self._best_x is not None:
            x = self._best_x.copy()
            sigma = 0.5
            while remaining > 0 and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped(y)
                remaining -= 1
                if fy < self._best_f:
                    x = y
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-4:
                        break

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
