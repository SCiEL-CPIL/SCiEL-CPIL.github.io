"""Rewritten optimizer using CMA-ES with adaptive restarts for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_evals, rng, deadline):
        """Run CMA-ES with a given evaluation budget and time deadline."""
        n = self.dim
        if n <= 0:
            return x0, np.inf, 0

        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc)
        cmu = (2 / ((n + 2) ** 2 + mueff)) * (min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cov = np.eye(n)
        evals = 0
        gen = 0

        # Evaluate initial point
        fx = func(x)
        evals += 1
        best_x = x.copy()
        best_f = fx

        while evals < max_evals and time.time() < deadline:
            gen += 1
            # Generate population
            offspring = np.empty((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                offspring[i] = x + sigma * (B @ (D * z))
            offspring = np.clip(offspring, -5, 5)

            # Evaluate offspring
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if evals >= max_evals or time.time() >= deadline:
                    break
                fvals[i] = func(offspring[i])
                evals += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = offspring[i].copy()

            if evals < max_evals and time.time() >= deadline:
                break

            if np.isnan(fvals).any() or np.isinf(fvals).any():
                break

            # Sort
            order = np.argsort(fvals)
            y_sorted = offspring[order]
            f_sorted = fvals[order]

            # Update mean
            y_mean = np.sum(weights[:mu, None] * y_sorted[:mu], axis=0)
            x_new = x + y_mean

            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (y_mean / sigma)

            # Update evolution path pc
            ps_norm = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - cs) ** (2 * gen))
            hsig = (ps_norm / (denom + 1e-10)) < (1 + cs) * chi_n * 1.4
            pc = (1 - cc) * pc + cc * (y_mean / sigma) * hsig

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            cov = (1 + c1 * delta_hsig - c1 - cmu) * cov
            cov += c1 * np.outer(pc, pc)
            for i in range(mu):
                yi = (y_sorted[i] - x) / sigma
                cov += cmu * weights[i] * np.outer(yi, yi)

            # Update step size (sigma)
            sigma *= np.exp((cs / damps) * (ps_norm / chi_n - 1))

            # Update eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                D = np.sqrt(np.maximum(eigvals, 1e-20))
                B = eigvecs
            except np.linalg.LinAlgError:
                break

            x = x_new

            if sigma < 1e-15:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Leave 20s safety margin
        rng = np.random.default_rng(42)
        n = self.dim

        best_x = None
        best_f = np.inf
        total_evals = 0

        # Multi-start CMA-ES with different initializations
        n_restarts = 4
        evals_per_restart = self.budget // n_restarts

        # Different starting strategies
        starts = [
            (np.zeros(n), 2.0),
            (rng.uniform(-5, 5, n), 2.0),
            (rng.uniform(-3, 3, n), 1.5),
            (np.ones(n) * 1.5, 1.0),
        ]

        for r in range(n_restarts):
            if total_evals >= self.budget or time.time() >= deadline:
                break

            x0, sigma0 = starts[r]
            remaining = min(self.budget - total_evals, max(50, evals_per_restart))
            x_best, f_best, evals_used = self._cma_es(func, x0, sigma0, remaining, rng, deadline)
            total_evals += evals_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we've found a very good solution, break early
            if best_f < 1e-8:
                break

        # Local refinement with Nelder-Mead if budget remains
        if best_x is not None and total_evals < self.budget - 50 and time.time() < deadline:
            remaining = self.budget - total_evals
            try:
                from scipy.optimize import minimize
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': min(remaining // 2, 300), 'xatol': 1e-10, 'fatol': 1e-12})
                total_evals += min(remaining // 2, 300)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Final CMA-ES refinement from best point
        if best_x is not None and total_evals < self.budget - 20 and time.time() < deadline:
            remaining = self.budget - total_evals
            x_best2, f_best2, evals_used = self._cma_es(func, best_x, 0.3, remaining, rng, deadline)
            total_evals += evals_used
            if f_best2 < best_f:
                best_f = f_best2
                best_x = x_best2

        if best_x is None:
            x0 = np.zeros(n)
            best_x = x0
            best_f = func(x0)
            total_evals += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
