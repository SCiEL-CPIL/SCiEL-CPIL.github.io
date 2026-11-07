"""CMA-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for box-constrained continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
            return val

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 16  # Population size
        sigma0 = 2.2  # Initial step size
        cc = (4 + dim / lambda_) / (dim + 4.3)
        cs = (lambda_ + 2 + 2 / (dim + 1)) / (dim + lambda_ + 5)
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c1 = (2 + 2 / (mu_eff - 1)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))

        # Initial population centered at origin
        rng = np.random.default_rng(42)
        x0 = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Generate initial population
        def generate_population(sigma):
            pop = []
            for _ in range(lambda_):
                z = rng.standard_normal(dim)
                x = x0 + sigma * z
                x = np.clip(x, -5, 5)
                pop.append(x)
            return np.array(pop)

        # Main CMA-ES loop
        sigma = sigma0
        try:
            while calls < budget and time.time() - start_time < time_limit:
                # Generate and evaluate population
                pop = generate_population(sigma)
                fvals = np.array([safe_call(x) for x in pop])

                # Sort by fitness
                idx = np.argsort(fvals)
                pop_sorted = pop[idx]
                fvals_sorted = fvals[idx]

                # Update best
                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = pop_sorted[0].copy()

                # Weighted recombination
                xmean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

                # Update evolution path
                C_sqrt_inv = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_sqrt_inv = np.linalg.inv(C_sqrt_inv)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv.T @ (xmean - x0) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 + (1 - cs) ** (2 * (calls + 1)))

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (xmean - x0) / sigma

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) - C)
                C += cmu * np.sum(weights[:, None, None] * np.outer(pop_sorted[:mu] - x0, pop_sorted[:mu] - x0), axis=0)

                # Ensure positive semi-definite
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-15)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size (CSA)
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x0 = xmean
                x0 = np.clip(x0, -5, 5)

        except (TimeoutError, Exception):
            pass

        # Final local refinement with (1+1)-ES if budget remains
        remaining = budget - calls
        if remaining > 10 and best_x is not None and time.time() - start_time < time_limit - 5:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.1
            rng_local = np.random.default_rng(123)

            for _ in range(remaining):
                if time.time() - start_time > time_limit:
                    break
                y = x + sigma_local * rng_local.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = safe_call(y)
                except (TimeoutError, Exception):
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9

                if sigma_local < 1e-8:
                    break

            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
