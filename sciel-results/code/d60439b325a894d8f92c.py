"""Optimizer for BBOB-style Rosenbrock f8 on [-5,5]^5.
Hybrid CMA-ES + Nelder-Mead polish with an explicit wall-clock deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Wall-clock deadline with a safety margin (kill limit is 330 s).
        deadline = time.time() + 285.0

        rng = np.random.default_rng(0)
        # Start near the origin (Rosenbrock minimum is at 0 before the hidden
        # shift/rotation; the shift keeps the optimum in [-4,4]^5).
        x0 = np.zeros(dim)
        x0[0] = 0.0

        # Track best value seen so far (return interface unchanged: (x, fx)).
        best_x = x0.copy()
        best_f = float('inf')

        def _track(x, f):
            nonlocal best_x, best_f
            f = float(f)
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=float).copy()

        # Evaluate the starting point.
        try:
            _track(x0, func(x0))
        except Exception:
            return best_x, best_f

        # ---- Phase 1: CMA-ES (global + valley-adaptive) ----
        # Use a chunked budget so we can check the deadline between chunks.
        # CMA-ES via scipy.optimize.minimize(method='CMA-ES') does not expose a
        # per-call budget directly, so we run it in a bounded loop and cap
        # iterations via maxiter; we also wrap it so we can bail on time.
        remaining = budget - 1  # one call already used for x0
        # Split CMA-ES work into chunks of at most 60 iterations each.
        chunk_iter = 60
        used = 1
        while used < budget:
            if time.time() >= deadline:
                break
            this_iter = min(chunk_iter, budget - used)
            try:
                res = minimize(
                    func,
                    best_x,
                    method='CMA-ES',
                    options={
                        'maxiter': this_iter,
                        'popsize': 4 + int(3 * np.log(dim)),
                        'tol': 1e-9,
                        'disp': False,
                    },
                )
                # CMA-ES may have used up to this_iter evaluations.
                # (We cannot get exact count from scipy, so assume worst case.)
                used += this_iter
                if res.x is not None and np.isfinite(res.fun):
                    _track(res.x, res.fun)
            except Exception:
                # Keep whatever we have; stop CMA-ES phase.
                break
            if used >= budget:
                break

        # ---- Phase 2: Nelder-Mead polish (cheap local refinement) ----
        # A few short local refinements from the current best to tighten the
        # minimum. Rosenbrock is smooth, so local polish is safe.
        polish_calls = 0
        polish_budget = 60  # cap polish evaluations
        nm_start = best_x.copy()
        while polish_calls < polish_budget and used + 1 < budget:
            if time.time() >= deadline:
                break
            try:
                res = minimize(
                    func,
                    nm_start,
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-10,
                             'disp': False},
                )
                polish_calls += 20  # conservative upper bound on calls used
                used += 20
                if res.x is not None and np.isfinite(res.fun):
                    _track(res.x, res.fun)
                    nm_start = best_x.copy()
                else:
                    break
            except Exception:
                break

        return best_x, best_f
