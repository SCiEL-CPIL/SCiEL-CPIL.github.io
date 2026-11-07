"""Random-restart + Powell + L-BFGS-B optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Robust continuous minimizer for the Bent Cigar (f12) on [-5, 5]^5.

    Strategy (structural, tuned for the anisotropic valley):
      1. A handful of cheap random restarts to locate the basin despite the
         hidden shift/rotation.
      2. Powell's method: adapts its conjugate directions to the narrow
         valley, which is far more effective than Nelder-Mead here.
      3. L-BFGS-B: gradient-based polish of the best basin found.
      4. Per-axis golden-section refinement on the narrow axes.
    Every func call is guarded by a call-budget and an explicit wall-clock
    deadline; a late exception keeps the calls already made.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin vs 300s hard limit
        dim = self.dim
        budget = self.budget

        state = {"calls": 0, "best_f": np.inf, "best_x": None}

        def evaluate(x):
            if state["calls"] >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(dim)
            fx = func(x)
            state["calls"] += 1
            if fx < state["best_f"]:
                state["best_f"] = fx
                state["best_x"] = x.copy()
            return fx

        def remaining():
            return budget - state["calls"]

        # --- Phase 1: cheap random restarts to find the basin ---
        rng = np.random.default_rng(12345)
        candidates = [np.zeros(dim)]
        for _ in range(5):
            candidates.append(rng.uniform(-4.0, 4.0, size=dim))

        for cand in candidates:
            if remaining() <= 25 or time.time() > deadline - 5:
                break
            f = evaluate(cand)
            if f == np.inf:
                continue

            # Powell from this candidate (adaptive directions for the valley)
            def powell_func(x):
                if state["calls"] >= budget or time.time() > deadline - 3:
                    return 1e12
                v = evaluate(x)
                return 1e12 if v == np.inf else v

            try:
                minimize(powell_func, cand, method='Powell',
                         bounds=[(-5.0, 5.0)] * dim,
                         options={'maxiter': 40, 'xtol': 1e-9, 'ftol': 1e-12})
            except Exception:
                pass

        if state["best_x"] is None:
            return (np.zeros(dim), np.inf)

        # --- Phase 2: L-BFGS-B gradient polish of the best basin ---
        if remaining() > 20 and time.time() < deadline - 5:
            def lbfgs_func(x):
                if state["calls"] >= budget or time.time() > deadline - 3:
                    return 1e12
                v = evaluate(x)
                return 1e12 if v == np.inf else v

            try:
                minimize(lbfgs_func, state["best_x"].copy(), method='L-BFGS-B',
                         bounds=[(-5.0, 5.0)] * dim,
                         options={'maxiter': 40, 'ftol': 1e-14, 'gtol': 1e-10})
            except Exception:
                pass

        # --- Phase 3: per-axis golden-section refinement (narrow axes) ---
        if remaining() > 15 and time.time() < deadline - 5:
            cur = state["best_x"].copy()
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            inv = 1.0 / phi
            for _ in range(2):
                for i in range(dim):
                    if state["calls"] >= budget - 6 or time.time() > deadline - 3:
                        break
                    a = max(-5.0, cur[i] - 1.0)
                    b = min(5.0, cur[i] + 1.0)
                    if b - a < 1e-9:
                        continue
                    c = b - inv * (b - a)
                    d = a + inv * (b - a)
                    fc = evaluate(cur.copy() * 0 + np.concatenate((cur[:i], [c], cur[i + 1:])))
                    fd = evaluate(np.concatenate((cur[:i], [d], cur[i + 1:])))
                    for _ in range(24):
                        if state["calls"] >= budget - 6 or time.time() > deadline - 3:
                            break
                        if fc < fd:
                            b, d, fd = d, c, fc
                            c = b - inv * (b - a)
                            fc = evaluate(np.concatenate((cur[:i], [c], cur[i + 1:])))
                        else:
                            a, c, fc = c, d, fd
                            d = a + inv * (b - a)
                            fd = evaluate(np.concatenate((cur[:i], [d], cur[i + 1:])))
                        if abs(b - a) < 1e-9:
                            break
                    cur[i] = (a + b) / 2.0
                    evaluate(cur)

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
