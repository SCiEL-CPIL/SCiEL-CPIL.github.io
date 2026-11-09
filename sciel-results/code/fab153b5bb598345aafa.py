"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based stochastic search for BBOB-style convex functions.

    Strategy:
      1. Cheap LHS spread to establish a good early best (AOCC rewards early progress).
      2. Nelder-Mead local refinement from the best sample.
      3. CMA-ES population search from the best sample (rotation-adaptive).
      4. Final L-BFGS-B polish on the convex function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # ------------------------------------------------------------------
        # Wrap func with a call counter so we can track budget usage
        # ------------------------------------------------------------------
        call_count = [0]

        def wrapped(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            return float(func(x))

        # ------------------------------------------------------------------
        # Phase 1: Latin Hypercube spread for a strong early best
        # ------------------------------------------------------------------
        n_lhs = 31
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + (perm + 0.5) * (10.0 / n_lhs)
        # Explicitly include origin (Sphere minima after shift often near center)
        lhs_points[0] = np.zeros(dim)

        best_x = None
        best_f = np.inf
        for pt in lhs_points:
            f = wrapped(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # ------------------------------------------------------------------
        # Phase 2: Nelder-Mead local refinement from the best LHS point
        # ------------------------------------------------------------------
        if time.time() < deadline - 60 and call_count[0] < budget - 100:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="Nelder-Mead",
                               options={"maxiter": 80, "xatol": 1e-7, "fatol": 1e-9})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase 3: CMA-ES population search (rotation-adaptive)
        # ------------------------------------------------------------------
        if time.time() < deadline - 30 and call_count[0] < budget - 50:
            try:
                from cma import CMA
                # Set initial sigma based on problem scale
                sigma0 = 2.0
                opts = {
                    "maxiter": max(1, (budget - call_count[0]) // 5),
                    "popsize": max(4 + int(3 * np.log(dim)), 6),
                    "CMA_adaptation_of_diagonal": True,
                    "bounds": [(-5.0,) * dim, (5.0,) * dim],
                    "verbose": -9,
                    "seed": 42,
                }
                es = CMA(best_x, sigma0, opts)
                while not es.stop() and call_count[0] < budget - 20 and time.time() < deadline - 10:
                    solutions = es.ask()
                    for sol in solutions:
                        if call_count[0] >= budget - 20 or time.time() >= deadline - 10:
                            break
                        f = wrapped(sol)
                        es.tell(sol, f)
                # Update best from CMA-ES
                if es.best.f < best_f:
                    best_f = float(es.best.f)
                    best_x = np.clip(es.best.x, -5.0, 5.0).copy()
            except ImportError:
                # Fallback: simple population-based search if cma not available
                self._population_search(wrapped, best_x, best_f, budget, dim, call_count, deadline)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase 4: L-BFGS-B final polish (convex Sphere function)
        # ------------------------------------------------------------------
        if time.time() < deadline - 5 and call_count[0] < budget - 5:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped, best_x, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 50, "ftol": 1e-12, "gtol": 1e-8})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f

    def _population_search(self, wrapped, best_x, best_f, budget, dim, call_count, deadline):
        """Fallback population-based search when cma module is unavailable."""
        rng = np.random.default_rng(123)
        pop_size = 20
        sigma = 2.0
        pop = [best_x + sigma * rng.standard_normal(dim) for _ in range(pop_size)]
        pop_f = [wrapped(np.clip(p, -5, 5)) for p in pop]

        for _ in range(budget - call_count[0] - 10):
            if call_count[0] >= budget - 10 or time.time() >= deadline - 5:
                break
            # Select best
            idx = int(np.argmin(pop_f))
            if pop_f[idx] < best_f:
                best_f = pop_f[idx]
                best_x = np.clip(pop[idx], -5, 5).copy()
            # Generate offspring around best
            child = best_x + sigma * rng.standard_normal(dim)
            child = np.clip(child, -5, 5)
            fc = wrapped(child)
            if fc < pop_f[idx]:
                pop[idx] = child
                pop_f[idx] = fc
            # Adapt sigma
            if fc < best_f:
                sigma *= 1.3
            else:
                sigma *= 0.9
# EVOLVE-BLOCK-END
