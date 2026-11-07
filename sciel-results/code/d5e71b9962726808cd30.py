"""Rewritten optimizer: CMA-ES with adaptive restarts and time guard."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts on the BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        best_x = np.full(n, np.nan)
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=float).reshape(n)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            return val

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_run, rng):
            """Run CMA-ES for budget_run evaluations starting from x0 with step sigma0."""
            nonlocal best_x, best_f, calls_made

            n = self.dim
            m = max(4, int(4 + 3 * np.log(n)))  # population size
            # Standard CMA-ES parameters
            cm = 1. / (n + 4)
            cs = (n + 1.3) / (n + 3 + 2 * n / (n + 4))
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
            cc = (4 + n / (n + 4)) / ((n + 4) + 4 + n / (n + 4))
            mu = m // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1. / np.sum(weights ** 2)
            c1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (cm - 1) + 2 * cm / (n + 1)
            cmu = (2 / ((n + 2) ** 2 + mu_eff)) * (min(cm, 1 - c1) - 1)

            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

            f_hist = []
            x_hist = [x0.copy()]
            f_best = np.inf

            for gen in range(budget_run // m + 1):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                # Sample population
                B = rng.standard_normal((m, n))
                # Apply covariance
                L = np.linalg.cholesky(C)
                pop = xmean[None, :] + sigma * (B @ L.T)
                pop = np.clip(pop, -5.0, 5.0)

                # Evaluate
                fvals = []
                for i in range(m):
                    if calls_made >= total_budget or time.time() > deadline:
                        fvals.append(np.inf)
                        continue
                    fvals.append(safe_func(pop[i]))
                    fvals_arr = np.array(fvals)
                    if np.isfinite(fvals_arr).any():
                        cur_best = np.min(fvals_arr[np.isfinite(fvals_arr)])
                        if cur_best < f_best:
                            f_best = cur_best
                            idx = np.argmin(fvals_arr)
                            f_best = fvals_arr[idx]
                            x_hist.append(pop[idx].copy())
                        if cur_best < best_f:
                            best_f = cur_best
                            best_x = pop[idx].copy()
                    fvals = fvals_arr.tolist()

                if not np.isfinite(fvals).any():
                    break

                fvals_arr = np.array(fvals)
                order = np.argsort(fvals_arr)
                fvals_sorted = fvals_arr[order]
                pop_sorted = pop[order]

                # Update best
                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = pop_sorted[0].copy()

                # Weighted mean of best mu
                xnew = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

                # Update covariance matrix
                y = (xnew - xmean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cs) ** 2) * np.eye(n)) \
                    + cmu * np.sum(weights[:, None, None] * np.outer(pop_sorted[:mu] - xmean, pop_sorted[:mu] - xmean), axis=0) / sigma ** 2

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                # Update step size
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) / chi_n * y
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 100.0)

                # Update mean
                pc = (1 - cc) * pc + cc * (1 - mu_eff / mu_eff) * y  # simplified
                xmean = xnew

                # Check for convergence
                if np.all(np.abs(eigvals) < 1e-14) or sigma < 1e-14:
                    break

            return xmean, f_best

        # Run with restarts
        rng = np.random.default_rng(42)
        # First run: standard CMA-ES from center
        x0 = np.zeros(n)
        sigma0 = 2.0
        budget_per_run = min(total_budget, 300)

        # Try a few initial points with different strategies
        initial_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            rng.uniform(-4, 4, n),
            rng.uniform(-4, 4, n),
        ]

        # Use remaining budget across restarts
        used = calls_made
        for i, x0 in enumerate(initial_points):
            if used >= total_budget or time.time() > deadline:
                break

            remaining = total_budget - used
            if remaining < 20:
                break

            # Allocate budget: first run gets more
            if i == 0:
                budget_run = min(remaining, 200)
            else:
                budget_run = min(remaining, 80)

            sigma0 = 2.0 if i == 0 else 1.0

            cma_es(x0, sigma0, budget_run, rng)
            used = calls_made

        return best_x, best_f
