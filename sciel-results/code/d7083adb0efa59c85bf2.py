import time
import numpy as np
from scipy.optimize import minimize

N = 100
W = 2.0 / N
DEADLINE = 110.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _overlap(h, n_shift=120, n_x=120):
    h = np.asarray(h, float)
    ts = np.linspace(0.0, 2.0, n_shift)
    best = 0.0
    for t in ts:
        hi = 2.0 - t
        if hi <= 0:
            continue
        xs = np.linspace(0.0, hi, n_x)
        ih = np.clip((xs / W).astype(np.int64), 0, N-1)
        ixt = np.clip(((xs + t) / W).astype(np.int64), 0, N-1)
        ov = np.trapz(h[ih] * (1.0 - h[ixt]), xs) / 2.0
        if ov > best:
            best = ov
    return best

def _project(h):
    # clip to [0,1] then water-fill to make sum exactly N/2
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(50):
        s = h.sum()
        d = (target - s) / N
        if abs(d) < 1e-12:
            break
        h2 = np.clip(h + d, 0.0, 1.0)
        if np.allclose(h2, h):
            break
        h = h2
    # exact fix on free (non-saturated) entries
    s = h.sum()
    d = target - s
    free = (h > 1e-9) & (h < 1.0 - 1e-9)
    if d != 0 and free.any():
        h[free] = np.clip(h[free] + d / free.sum(), 0.0, 1.0)
    return h

def _fourier_init(modes=6, amp=0.3):
    x = np.linspace(0.0, 2.0, N, endpoint=False) + W/2
    h = np.full(N, 0.5)
    for m in range(1, modes+1):
        h += amp/m * np.cos(2*np.pi*m*x/2.0)
        h += amp/m * np.sin(2*np.pi*m*x/2.0)
    return _project(h)

def construct():
    best_h = _fourier_init()
    best_v = _overlap(best_h)

    def obj(h):
        return _overlap(h)

    cons = [{'type': 'eq', 'fun': lambda h: h.sum() - N/2.0}]
    bounds = [(0.0, 1.0)] * N

    inits = [best_h]
    rng = np.random.default_rng(0)
    for _ in range(3):
        inits.append(_project(0.5 + 0.4*rng.standard_normal(N)))

    for h0 in inits:
        if _elapsed() > DEADLINE - 20:
            break
        try:
            res = minimize(obj, h0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 60, 'ftol': 1e-8})
            if res.success or res.fun < best_v:
                h = _project(res.x)
                v = _overlap(h)
                if v < best_v:
                    best_v = v
                    best_h = h
        except Exception:
            pass

    return best_h
