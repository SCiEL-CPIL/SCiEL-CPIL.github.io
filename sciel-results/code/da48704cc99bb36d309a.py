"""Second autocorrelation inequality: maximize value = l2/(norm_1*norm_inf)."""
import time
import numpy as np

N = 50

def _eval(h):
    """Exact evaluator (mirrors the public contract)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    return l2 / (norm_1 * norm_inf)

def _profile(a, b):
    """Smooth concentrated (unimodal) profile, nonnegative."""
    x = np.linspace(0.0, 1.0, N)
    return a * np.exp(-b * (x - 0.5) ** 2)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under the 120s limit

    # ---- cheap grid search over (a, b) ----
    best_val = -1.0
    best_h = np.ones(N)
    a_grid = np.linspace(1.0, 30.0, 60)
    b_grid = np.linspace(0.5, 60.0, 90)
    for a in a_grid:
        for b in b_grid:
            h = _profile(a, b)
            v = _eval(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
        if time.time() > deadline:
            return list(best_h)

    # ---- local refinement (Nelder-Mead) around the best grid point ----
    def _neg(params):
        a, b = params
        return -_eval(_profile(a, b))

    try:
        from scipy.optimize import minimize
        # locate current best (a,b) by rescanning the grid cheaply
        cur_a, cur_b = float(a_grid[0]), float(b_grid[0])
        cv = _neg([cur_a, cur_b])
        for a in a_grid:
            for b in b_grid:
                v = _neg([a, b])
                if v < cv:
                    cv = v
                    cur_a, cur_b = float(a), float(b)
        res = minimize(_neg, [cur_a, cur_b], method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-9,
                                "adaptive": True})
        if res.x[0] > 0 and res.x[1] > 0:
            h = _profile(res.x[0], res.x[1])
            v = _eval(h)
            if v > best_val:
                best_val = v
                best_h = h
    except Exception:
        pass

    # ---- per-coordinate coordinate ascent (fast, robust) ----
    def _step(h):
        h = np.asarray(h, dtype=float)
        g = np.convolve(h, h)
        m = len(g)
        y = np.concatenate(([0.0], g, [0.0]))
        w = 1.0 / (m + 1)
        l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
        n1 = np.sum(np.abs(g)) / (m + 1)
        ni = np.max(np.abs(g))
        return l2, n1, ni

    def _val(h):
        l2, n1, ni = _step(h)
        return l2 / (n1 * ni)

    for _ in range(40):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            orig = best_h[i]
            for f in (1.15, 0.87, 1.3, 0.7, 1.5, 0.5, 1.8, 0.3):
                cand = best_h.copy()
                cand[i] = orig * f
                v = _val(cand)
                if v > best_val:
                    best_val = v
                    best_h = cand
                    improved = True
                    break
        if not improved:
            break

    return list(best_h)
