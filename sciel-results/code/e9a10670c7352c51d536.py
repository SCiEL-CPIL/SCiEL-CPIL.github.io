"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Model-based optimizer: Sobol global sampling + Nelder-Mead local refinement.

    Strategy:
      1. Generate a deterministic 20-point Sobol sequence mapped into [-5, 5]^d.
      2. Evaluate all points (recorded by the evaluator).
      3. Run Nelder-Mead simplex local refinement from the best point.
      4. Guard every function call against a wall-clock deadline so the program
         always returns before the 300 s limit (killed at 330 s).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # 30 s safety margin before 300 s / 330 s kill
        d = self.dim
        budget = self.budget

        # --- Deadline-guarded wrapper ----------------------------------------
        state = {"count": 0, "ok": True}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300
            v = func(np.asarray(x, dtype=float))
            state["count"] += 1
            try:
                v = float(v)
            except Exception:
                return 1e300
            if not np.isfinite(v):
                return 1e300
            return v

        # --- 1. Sobol global sample ------------------------------------------
        n_sobol = min(20, budget)
        if n_sobol >= 2:
            sampler = qmc.Sobol(d, scramble=False)
            sob = sampler.random(n_sobol)  # in [0, 1)^d
            pts = -5.0 + 10.0 * sob
        else:
            pts = np.array([np.zeros(d)])

        best_x = None
        best_f = np.inf
        for p in pts:
            if not state["ok"]:
                break
            f = wf(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # --- 2. Nelder-Mead local refinement ---------------------------------
        if best_x is not None and state["ok"]:
            res = minimize(
                wf,
                best_x,
                method="Nelder-Mead",
                options={
                    "maxiter": 5000,
                    "maxfev": max(1, budget - state["count"] - 1),
                    "xatol": 1e-8,
                    "fatol": 1e-8,
                    "adaptive": True,
                },
            )
            if res is not None:
                xf = res.x
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff) and ff < best_f:
                    best_x = np.clip(np.asarray(xf, dtype=float), -5.0, 5.0)
                    best_f = ff

        return best_x, best_f
# EVOLVE-BLOCK-END
