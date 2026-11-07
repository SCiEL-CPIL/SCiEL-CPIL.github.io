"""Optimizer for BBOB-style rotated/shifted Rosenbrock (f8), dim 5, box [-5,5].

Strategy: global basin-of-attraction coverage (L-BFGS-B from diverse starts)
-> CMA-ES to refine basin choice -> L-BFGS-B + (1+1)-ES local polish.
A wall-clock deadline guard returns the best found before the 300 s kill.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.monotonic()
        deadline = t0 + 285.0          # safety margin before the 300 s kill

        # ---- evaluation wrapper: counts, guards, never raises ----
        state = {"n": 0}
        best = {"x": None, "f": None}

        def wrap(x):
            if state["n"] >= budget:
                # out of budget: return a cached value (no new eval)
                return best["f"] if best["f"] is not None else 0.0
            if time.monotonic() > deadline:
                return best["f"] if best["f"] is not None else 0.0
            try:
                xv = np.asarray(x, dtype=float).reshape(dim)
                xv = np.clip(xv, -5.0, 5.0)
            except Exception:
                return best["f"] if best["f"] is not None else 0.0
            try:
                fv = float(func(xv))
            except Exception:
                return best["f"] if best["f"] is not None else 0.0
            if not np.isfinite(fv):
                return best["f"] if best["f"] is not None else 0.0
            state["n"] += 1
            if best["f"] is None or fv < best["f"]:
                best["f"] = fv
                best["x"] = xv.copy()
            return fv

        def done():
            return state["n"] >= budget or time.monotonic() > deadline

        # ---- Phase 1: global basin coverage via L-BFGS-B from diverse starts ----
        rng = np.random.default_rng(0)
        starts = [np.full(dim, 0.0)]
        # Hammersley quasi-random points (low-discrepancy, diverse)
        nq = 8
        for k in range(1, nq + 1):
            v = np.array([(k * (2 * i + 1)) % 1 for i in range(dim)])
            starts.append(v * 10.0 - 5.0)
        # box corners (extreme basin probes)
        for c in [[-5.0] * dim, [5.0] * dim,
                  [-5.0, 5.0, -5.0, 5.0, -5.0],
                  [5.0, -5.0, 5.0, -5.0, 5.0]]:
            starts.append(np.array(c, dtype=float))
        # a few random points
        for _ in range(3):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        global_pts = []
        for s in starts:
            if done():
                break
            try:
                res = minimize(wrap, s, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 120, "maxfun": 60})
                if res.x is not None:
                    global_pts.append((float(res.fun), np.clip(np.asarray(res.x, float), -5, 5)))
            except Exception:
                pass

        # pick the best global point
        if global_pts:
            global_pts.sort(key=lambda t: t[0])
            x0 = global_pts[0][1].copy()
            f0 = global_pts[0][0]
        else:
            x0 = np.full(dim, 0.0)
            f0 = wrap(x0)

        # ---- Phase 2: CMA-ES global refinement (basin choice + valley entry) ----
        if not done() and state["n"] < budget - 20:
            try:
                from cma import CMA, CMAOptions
                vals = [p[0] for p in global_pts]
                pts = np.array([p[1] for p in global_pts]) if len(global_pts) > 1 else None
                spread = float(np.std(pts, axis=0).max()) if pts is not None and len(pts) > 1 else 0.0
                sigma0 = max(0.5, min(2.5, spread * 0.5 if spread > 0 else 1.0))
                pop = max(10, 4 + int(3 * np.log(dim)))
                # adaptive sigma floor: keep global moves meaningful in the valley
                def sigma_adapt(c, sigma, i):
                    s = sigma * (1.0 + 0.3 * np.exp(-i / 20.0))
                    return max(s, 0.25)
                opts = CMAOptions()
                opts["popsize"] = pop
                opts["maxfevals"] = max(10, min(budget - state["n"] - 15, 180))
                opts["bounds"] = [(-5.0, 5.0)] * dim
                opts["seed"] = 0
                opts["verbose"] = -9
                opts["tolfun"] = 1e-12
                opts["sigma_adapt"] = sigma_adapt
                es = CMA(x0, sigma0, opts)
                while not es.stop() and not done():
                    es.ask()
                    es.tell(es.xs, [wrap(x) for x in es.xs])
                if es.result.x is not None:
                    xb = np.clip(np.asarray(es.result.x, float), -5, 5)
                    fb = wrap(xb)
                    if best["f"] is None or fb < best["f"]:
                        best["f"], best["x"] = fb, xb.copy()
            except Exception:
                pass

        # ---- Phase 3: local polish ----
        xcur = best["x"].copy() if best["x"] is not None else x0
        fcur = best["f"] if best["f"] is not None else f0

        # L-BFGS-B polish from the current best
        if not done() and state["n"] < budget - 8:
            try:
                res = minimize(wrap, xcur, method="L-BFGS-B",
                               bounds=[(-5.0, 5.0)] * dim,
                               options={"maxiter": 150, "maxfun": 40})
                if res.x is not None:
                    xb = np.clip(np.asarray(res.x, float), -5, 5)
                    fb = wrap(xb)
                    if best["f"] is None or fb < best["f"]:
                        best["f"], best["x"] = fb, xb.copy()
            except Exception:
                pass

        # tight (1+1)-ES polish on the valley
        if best["x"] is not None and not done() and state["n"] < budget - 2:
            x = best["x"].copy()
            fx = best["f"]
            sigma = 0.15
            while state["n"] < budget and time.monotonic() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = wrap(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
            if fx < best["f"]:
                best["f"], best["x"] = fx, x.copy()

        # ensure at least one evaluation happened (avoid zero-call invalidity)
        if state["n"] == 0:
            wrap(np.full(dim, 0.0))

        return best["x"], best["f"]
# EVOLVE-BLOCK-END
