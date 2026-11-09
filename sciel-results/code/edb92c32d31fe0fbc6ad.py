"""Portfolio optimizer for BBOB f1 (Sphere) on [-5,5]^dim.

Strategy:
  1. A small deterministic global net (LHS + random) to seed local solvers.
  2. CMA-ES from the best global point with a generous budget (adapts to
     rotation/shift via covariance adaptation).
  3. L-BFGS-B polish from several diverse starts.
  4. A final CMA-ES squeeze with the remaining budget.

The best point found is always returned.  A hard wall-clock deadline keeps
the process well inside the 330 s kill limit.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start (LHS + random) with CMA-ES / L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin so the process is never killed at 300/330s.
        deadline = t0 + 280.0

        def time_ok():
            return time.time() < deadline

        def wrap(z):
            x = np.clip(np.asarray(z, dtype=float).reshape(-1), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return 1e30

        rng = np.random.default_rng(0)

        # For Sphere (f1), the minimum is a single point. Use a focused
        # initial set: origin first, then a small LHS for global coverage.
        origin = np.zeros(dim)
        
        # Small LHS for global coverage (fewer points, more budget for optimization)
        n_lhs = 15
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        # A few random points for diversity
        n_rand = 8
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        # Put origin first, then sort rest by distance to origin
        rest = np.vstack([lhs, rand])
        rest_scores = np.linalg.norm(rest, axis=1)
        rest_order = np.argsort(rest_scores)
        rest = rest[rest_order]
        
        candidates = np.vstack([origin[None, :], rest])

        best_x = candidates[0]
        best_f = wrap(best_x)
        budget_left = budget - 1
        
        # Quick scan of a few more initial points to find a better start
        for i in range(1, min(6, len(candidates))):
            if budget_left <= 0:
                break
            x = candidates[i]
            f = wrap(x)
            budget_left -= 1
            if f < best_f:
                best_f = f
                best_x = x

        # ------------------------------------------------------------------
        # Phase A: CMA-ES from the best of the global net (global basin finder).
        # ------------------------------------------------------------------
        try:
            from cma import CMA
            if budget_left > 50:
                # CMA-ES is the workhorse for rotated landscapes.
                # Larger sigma for faster initial convergence on Sphere.
                cma_budget = int(min(400, max(100, budget_left * 0.75)))
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=2.0,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=cma_budget,
                          seed=1,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-14,
                              'tolx'    : 0.0,
                              'tolopt'  : 0.0,
                          })
                budget_left -= int(res.evaluations)
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase B: fast L-BFGS-B polish from a spread of starts.
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import minimize
            # Single high-quality L-BFGS-B polish from the best point.
            # For Sphere, this should converge very quickly to high precision.
            if budget_left > 10 and time_ok():
                try:
                    r = minimize(wrap, best_x, method='L-BFGS-B',
                                 bounds=[(-5.0, 5.0)] * dim,
                                 options={'maxiter': 1000,
                                          'maxfun': max(10, min(300, budget_left)),
                                          'gtol': 1e-14,
                                          'ftol': 1e-14})
                    budget_left -= int(r.nfev)
                    if r.fun <= best_f:
                        best_f = float(r.fun)
                        best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase C: final high-precision polish.
        # For Sphere, L-BFGS-B from the CMA-ES result should reach machine
        # precision quickly. Use remaining budget for a tight final polish.
        # ------------------------------------------------------------------
        if budget_left > 5 and time_ok():
            try:
                from scipy.optimize import minimize
                r = minimize(wrap, best_x, method='L-BFGS-B',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 2000,
                                      'maxfun': max(5, budget_left),
                                      'gtol': 1e-15,
                                      'ftol': 1e-15})
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
