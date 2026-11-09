"""Combined optimizer: Latin Hypercube init + (1+1)-ES with adaptive sigma + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: LHS sampling, (1+1)-ES, and Nelder-Mead polish for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)

    def _latin_hypercube(self, n, dim, rng):
        """Generate n points using Latin Hypercube sampling in [lb, ub]^dim."""
        pts = np.empty((n, dim))
        for d in range(dim):
            order = rng.permutation(n)
            pts[order, d] = (np.arange(n) + rng.random(n)) / n
        return self.lb + pts * (self.ub - self.lb)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # --- Phase 1: Latin Hypercube Initialization ---
        n_init = 30
        X = self._latin_hypercube(n_init, dim, rng)
        y = np.array([func(x) for x in X])

        calls = n_init
        idx_best = np.argmin(y)
        x_best = X[idx_best].copy()
        f_best = y[idx_best]

        # --- Phase 2: (1+1)-ES with adaptive sigma ---
        # Reserve ~80 calls for Nelder-Mead polish
        es_budget = min(budget - calls, 390)
        sigma = 1.0  # Start with moderate step size
        x_es = x_best.copy()
        fx_es = f_best

        for _ in range(es_budget):
            if time.time() > deadline - 10.0:
                break

            y_es = np.clip(x_es + sigma * rng.standard_normal(dim), -5, 5)
            fy_es = func(y_es)
            calls += 1

            if fy_es < fx_es:
                x_es, fx_es = y_es, fy_es
                sigma *= 1.15
                if fx_es < f_best:
                    f_best = fx_es
                    x_best = x_es.copy()
            else:
                sigma *= 0.85

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 5.0)

        # --- Phase 3: Nelder-Mead Polish ---
        # Use remaining budget for fine-tuning
        if calls < budget and time.time() < deadline - 5.0:
            # First NM pass with moderate perturbation
            x_nm = x_best.copy()
            f_nm = f_best

            def nm_obj(x):
                nonlocal calls
                if time.time() > deadline - 1.0:
                    return f_nm
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls += 1
                return fx

            # Nelder-Mead with tight tolerances
            res = minimize(
                nm_obj, x_nm, method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'maxiter': 200,
                    'adaptive': True
                }
            )

            if res.fun < f_best and calls <= budget:
                x_best = np.clip(res.x, -5, 5)
                f_best = res.fun

            # Second NM pass with even tighter tolerances if budget allows
            if calls < budget and time.time() < deadline - 2.0:
                x_nm2 = x_best.copy()
                f_nm2 = f_best

                def nm_obj2(x):
                    nonlocal calls
                    if time.time() > deadline - 0.5:
                        return f_nm2
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    calls += 1
                    return fx

                res2 = minimize(
                    nm_obj2, x_nm2, method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 100,
                        'adaptive': True
                    }
                )

                if res2.fun < f_best and calls <= budget:
                    x_best = np.clip(res2.x, -5, 5)
                    f_best = res2.fun

        return x_best, f_best
# EVOLVE-BLOCK-END
