"""Optimize 50 nonnegative step heights to maximize the second-autocorrelation ratio."""
import time
import numpy as np

N = 50
DEADLINE = 150.0
SAFETY = 15.0  # seconds margin before the 150s kill

def _ratio(h):
    """Raw objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(np.abs(h)))
    if l1 <= 0:
        return -1e300
    conv = np.convolve(h, h, mode="full")
    sup = float(np.max(np.abs(conv)))
    if sup <= 0:
        return -1e300
    l2sq = float(np.dot(h, h))
    return l2sq / (l1 * sup)

def _neg_ratio(h):
    return -_ratio(h)

def _make_starts():
    """Diverse structured initializations (shape, ||h||_1=1)."""
    starts = []
    n = N
    # uniform
    starts.append(np.ones(n) / n)
    # single bump (Gaussian)
    for center in (n * 0.5, n * 0.3, n * 0.7):
        x = np.arange(n)
        g = np.exp(-((x - center) ** 2) / (2.0 * (n * 0.15) ** 2))
        starts.append(g / g.sum())
    # two-bump
    for (c1, c2) in ((n * 0.25, n * 0.75), (n * 0.2, n * 0.8)):
        x = np.arange(n)
        g = np.exp(-((x - c1) ** 2) / (2.0 * (n * 0.1) ** 2)) + \
            np.exp(-((x - c2) ** 2) / (2.0 * (n * 0.1) ** 2))
        starts.append(g / g.sum())
    # periodic / comb
    for period in (4, 5, 7, 10):
        p = np.array([1.0 if (i % period) < 2 else 0.0 for i in range(n)])
        if p.sum() > 0:
            starts.append(p / p.sum())
    # power-law
    for alpha in (0.5, 1.0, 2.0, 3.0):
        x = np.arange(1, n + 1, dtype=float)
        p = x ** (-alpha)
        starts.append(p / p.sum())
        p2 = (x.max() - x + 1) ** (-alpha)
        starts.append(p2 / p2.sum())
    # spike
    for k in range(0, n, 10):
        s = np.zeros(n); s[k] = 1.0
        starts.append(s)
    # random (fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = rng.random(n)
        r += 1e-9
        starts.append(r / r.sum())
    return starts

def _clip_nonneg(h):
    return np.clip(h, 0.0, None)

def _normalize_l1(h, target=1.0):
    s = float(np.sum(h))
    if s <= 0:
        return h
    return h * (target / s)

def _coordinate_descent(h, max_iter=400, tol=1e-9):
    """Cyclic coordinate ascent on the ratio (with ||h||_1 fixed to 1)."""
    h = _clip_nonneg(h).copy()
    h = _normalize_l1(h, 1.0)
    best = _ratio(h)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # try a few candidate values for coordinate i, keep the rest
            base = h.copy()
            cur = base[i]
            # candidate set: current, 0, and a few scalings of current
            cands = [0.0, cur]
            if cur > 0:
                cands += [cur * f for f in (0.5, 0.75, 1.25, 1.5, 2.0, 0.25, 3.0)]
            else:
                cands += [1e-4, 1e-3, 1e-2]
            best_val = best
            best_c = cur
            for c in cands:
                base[i] = c
                base = _normalize_l1(base, 1.0)
                v = _ratio(base)
                if v > best_val:
                    best_val = v
                    best_c = c
            if best_c != cur:
                h = base if best_c != cur else h
                # recompute h from the winning candidate
                base2 = h.copy()
                base2[i] = best_c
                h = _normalize_l1(base2, 1.0)
                improved = True
        if best_val > best:
            best = best_val
        if not improved:
            break
    return h, best

def _lbfgs_polish(h, time_budget):
    from scipy.optimize import minimize
    h0 = _clip_nonneg(h).copy()
    h0 = _normalize_l1(h0, 1.0)
    # parameterize with softplus to keep nonnegative; work in log-space
    # Use a transform: h_i = softplus(x_i), then normalize.
    def obj(x):
        h = 1.0 / (1.0 + np.exp(-x))
        h = _normalize_l1(h, 1.0)
        return -_ratio(h)
    x0 = np.log(np.clip(h0, 1e-6, None) * N)  # softplus(x0) ~ h0
    t_start = time.time()
    try:
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 2000, "maxfun": 20000,
                                "ftol": 1e-12, "gtol": 1e-9, "disp": False})
        if res.success or res.fun < obj(x0):
            hx = 1.0 / (1.0 + np.exp(-res.x))
            hx = _normalize_l1(hx, 1.0)
            return hx
    except Exception:
        pass
    return h0

def construct():
    t0 = time.time()
    def deadline():
        return (time.time() - t0) > (DEADLINE - SAFETY)

    starts = _make_starts()
    best_h = starts[0].copy()
    best_score = _ratio(best_h)

    for s in starts:
        if deadline():
            break
        # coordinate descent
        try:
            h_cd, sc_cd = _coordinate_descent(s)
            if sc_cd > best_score:
                best_score = sc_cd
                best_h = h_cd
        except Exception:
            pass
        if deadline():
            break
        # L-BFGS polish on the best so far
        try:
            h_pol = _lbfgs_polish(best_h, 20.0)
            sc_pol = _ratio(h_pol)
            if sc_pol > best_score:
                best_score = sc_pol
                best_h = h_pol
        except Exception:
            pass

    # Final high-precision polish pass if time remains
    if not deadline():
        try:
            h_final = _lbfgs_polish(best_h, 30.0)
            sc_final = _ratio(h_final)
            if sc_final > best_score:
                best_score = sc_final
                best_h = h_final
        except Exception:
            pass

    # Return nonnegative heights at the seed's L1 scale (sum = N) so the
    # raw value is on the same footing as the all-ones seed.
    out = _clip_nonneg(best_h)
    s = float(np.sum(out))
    if s <= 0:
        out = np.ones(N)
    else:
        out = out * (N / s)
    return [float(v) for v in out]
