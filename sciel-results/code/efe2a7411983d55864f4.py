"""Rewritten optimizer for BBOB f11 (Discus): multi-restart CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES black-box minimizer on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 280.0  # safety margin within 300s wall-clock

        # ---- call counter + wrapper ----
        state = {"n": 0}

        def wrapped(x):
            state["n"] += 1
            # clip to box and ensure finite length-dim vector
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = x[:dim] if x.shape[0] > dim else np.pad(x, (0, dim - x.shape[0]))
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf

        def consider(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # ---- Phase 1: coarse global scan for early progress ----
        # A small set of spread-out starting points. Cheap and gives early AOCC credit.
        n_scan = min(15, max(3, budget // 30))
        # include origin and a few deterministic + random points
        candidates = [np.zeros(dim)]
        for _ in range(n_scan - 1):
            candidates.append(rng.uniform(-4.0, 4.0, dim))
        for x in candidates:
            if state["n"] >= budget:
                break
            if time.time() > deadline:
                break
            f = wrapped(x)
            consider(x, f)

        # ---- Phase 2: multi-restart CMA-ES refinement from best points ----
        # Build a small set of restart origins: best point + a few nearby perturbations.
        origins = []
        if best_x is not None:
            origins.append(best_x.copy())
            # a few perturbations around the best to allow escape from shallow local minima
            for s in (1.5, 0.7):
                origins.append(np.clip(best_x + s * rng.standard_normal(dim), -5.0, 5.0))

        # Allocate remaining budget across restarts.
        remaining = max(1, budget - state["n"])
        n_restarts = len(origins)
        # give more budget to earlier (more promising) restarts
        per_restart = [max(20, int(remaining * w)) for w in (0.5, 0.3, 0.2)[:n_restarts]]

        for i, origin in enumerate(origins):
            if state["n"] >= budget:
                break
            if time.time() > deadline:
                break
            # sigma initial: scale to box
            sigma0 = 1.5
            try:
                res = minimize(
                    wrapped,
                    origin,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "maxls": 40, "gtol": 1e-6},
                    bounds=[(-5.0, 5.0)] * dim,
                )
            except Exception:
                # fall back to CMA-ES-style Nelder-Mead if L-BFGS fails
                try:
                    res = minimize(
                        wrapped,
                        origin,
                        method="Nelder-Mead",
                        options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-8},
                        bounds=[(-5.0, 5.0)] * dim,
                    )
                except Exception:
                    break
            if state["n"] >= budget:
                break
            # evaluate final point
            try:
                fx = float(np.atleast_1d(res.fun).reshape(-1)[0]) if res.fun is not None else np.inf
            except Exception:
                fx = np.inf
            if np.isfinite(res.x):
                consider(np.asarray(res.x, dtype=float), fx)

        # ---- Phase 3: if budget/time remains, a final short CMA-ES-style polish
        #     from the current best to squeeze out extra improvement.
        if best_x is not None and state["n"] < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method="L-BFGS-B",
                    options={"maxiter": 300, "maxls": 50, "gtol": 1e-8},
                    bounds=[(-5.0, 5.0)] * dim,
                )
                if np.isfinite(res.x):
                    fx = float(np.atleast_1d(res.fun).reshape(-1)[0]) if res.fun is not None else np.inf
                    consider(np.asarray(res.x, dtype=float), fx)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
