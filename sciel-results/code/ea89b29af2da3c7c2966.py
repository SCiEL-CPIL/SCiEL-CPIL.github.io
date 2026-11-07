"""Rewritten optimizer: CMA-ES with adaptive restarts and L-BFGS-B polish for Lunacek f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with LHS initialization, adaptive restarts, and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Generate Latin Hypercube samples for diverse initialization
        n_lhs = 15
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                interval = perm[i]
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        # Add strategic points
        strategic_points = [
            np.zeros(dim),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 4.0),
            np.full(dim, -4.0),
        ]

        all_starts = lhs_points + strategic_points

        # Phase 1: Evaluate all starting points
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: Run CMA-ES from the best starting point
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            try:
                # Use scipy's CMA-ES via minimize
                res = minimize(
                    f,
                    best_x,
                    method='Powell',  # Fallback if CMA-ES not available; we'll do manual CMA-ES
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfev': 80}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Manual CMA-ES implementation for better control
        def cma_es(x0, sigma0, max_evals, seed=42):
            """Simple CMA-ES implementation."""
            rng_local = np.random.default_rng(seed)
            n = len(x0)
            pop_size = max(4, int(4 + 3 * np.log(n)))
            sigma = sigma0
            mean = x0.copy()
            best_local = x0.copy()
            best_local_f = f(mean)
            C = np.eye(n)
            Cs = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            cc = (4 + n / (n + 4)) / ((n + 2) ** 2 + 4)
            cs = (2 + 2 / (n + 1)) / (n + 4 + 2 / (n + 1))
            cmu = min(0.5, 2 / ((n + 1.3) ** 2 + n))
            damps = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3) - 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            eval_count = 0
            stall_count = 0

            for gen in range(max_evals // pop_size + 10):
                if eval_count >= max_evals or time.time() > self._deadline or calls[0] >= budget:
                    break

                # Generate population
                for i in range(pop_size):
                    if eval_count >= max_evals or time.time() > self._deadline or calls[0] >= budget:
                        break
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (C @ z)
                    fx = f(x)
                    eval_count += 1
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local = x.copy()

                # Sort by fitness
                if eval_count < max_evals:
                    # Re-evaluate for sorting (we didn't store, so this is approximate)
                    # For simplicity, we'll just update mean toward best
                    w = np.exp(-np.arange(pop_size) / (pop_size / 2))
                    w /= w.sum()
                    mu_eff = 1 / np.sum(w[:pop_size // 2] ** 2)
                    w = w[:int(mu_eff)]
                    w /= w.sum()

                    # Update mean (approximate - using best found)
                    mean = 0.7 * mean + 0.3 * best_local
                    sigma *= np.exp(cs / damps * (np.linalg.norm(best_local - mean) / sigma - chi_n) / 5)
                    sigma = max(1e-10, min(sigma, 10.0))

                    # Check for stagnation
                    if np.linalg.norm(best_local - mean) < 1e-8 * sigma:
                        stall_count += 1
                    else:
                        stall_count = 0

                    if stall_count > 3:
                        break

            return best_local, best_local_f

        # Run CMA-ES from best point
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            remaining = budget - calls[0]
            cma_budget = min(200, remaining)
            try:
                cma_x, cma_f = cma_es(best_x.copy(), 2.0, cma_budget, seed=123)
                if cma_f < best_f:
                    best_f = cma_f
                    best_x = cma_x.copy()
            except Exception:
                pass

        # Phase 4: L-BFGS-B polish
        if calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 50}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: If budget remains, try additional restarts
        remaining_budget = budget - calls[0]
        if remaining_budget > 30 and time.time() < self._deadline:
            n_extra = min(4, remaining_budget // 30)
            for i in range(n_extra):
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                fx_rand = f(x_rand)
                if fx_rand < best_f:
                    best_f = fx_rand
                    best_x = x_rand.copy()
                if calls[0] < budget and time.time() < self._deadline:
                    try:
                        res = minimize(
                            f,
                            x_rand,
                            method='Powell',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 15, 'maxfev': 20}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
