"""Second autocorrelation inequality: maximize ||h||2^2 / (||h||1 * ||h*h||inf).

Structural change from seed: instead of fixing a large Sidon support with unit
heights, we (a) choose a SMALL support and (b) optimize the positive heights
numerically, since the objective is smooth in the heights. Fewer spikes make the
autoconvolution peak easy to control and let the optimizer concentrate mass where
it helps the L2/L1 balance.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin under the 150 s kill


def _autoconv(h):
    """Full linear autoconvolution (non-circular), h*h with h[i]h[i+k]."""
    return np.convolve(h, h, mode='full')


def _ratio(h):
    """||h||2^2 / (||h||1 * ||h*h||inf). Returns 0 for degenerate inputs."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 1e-300:
        return 0.0
    l2sq = float(np.dot(h, h))
    peak = float(np.max(np.abs(_autoconv(h))))
    if peak <= 1e-300:
        return 0.0
    return l2sq / (l1 * peak)


def _neg_ratio(h):
    return -_ratio(h)


def _optimize_heights(support):
    """Optimize positive heights on a fixed support to maximize the ratio."""
    k = len(support)
    if k == 0:
        return None, 0.0

    def obj(x):
        h = np.zeros(N)
        h[support] = x
        return -_ratio(h)

    x0 = np.ones(k)
    bounds = [(1e-9, None)] * k
    best_x, best_val = x0, obj(x0)
    for method in ('Nelder-Mead', 'Powell'):
        try:
            res = minimize(obj, x0, method=method, bounds=bounds,
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except Exception:
            pass
    h = np.zeros(N)
    h[support] = np.maximum(best_x, 0.0)
    val = _ratio(h)
    return h, val


def _random_supports(n_support, rng):
    """Yield candidate supports: small sets, including low-index clusters."""
    seen = set()
    # Always include the single-spike and a few structured small supports.
    for k in range(1, 8):
        for _ in range(40):
            s = sorted(rng.choice(N, size=k, replace=False).tolist())
            key = tuple(s)
            if key in seen:
                continue
            seen.add(key)
            yield s
    # Deterministic structured supports (arithmetic / geometric-ish)
    for start in range(0, N, 5):
        for step in (1, 2, 3, 5, 8):
            s = []
            p = start
            while p < N and len(s) < 7:
                s.append(p)
                p += step
            if len(s) >= 1:
                yield sorted(s)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio(best_h)

    # Evaluate a batch of supports; stop before the deadline.
    batch = 0
    for support in _random_supports(7, rng):
        if time.time() > DEADLINE:
            break
        h, val = _optimize_heights(support)
        if val > best_val:
            best_val = val
            best_h = h
        batch += 1
        if batch % 25 == 0 and time.time() > DEADLINE:
            break

    # Final safety: ensure we return something valid and non-degenerate.
    if best_val <= 0.0:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
