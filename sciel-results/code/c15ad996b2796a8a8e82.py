import numpy as np
import time

N = 100
TARGET = N / 2.0
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0


def _objective(h):
    """Exact objective: max over lags of the full cross-correlation h . (1-h)."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return float(np.max(corr))


def _project(h):
    """Project onto {0<=h<=1, sum(h)=N/2} by clip + iterative redistribution."""
    h = np.asarray(h, dtype=float).copy()
    for _ in range(200):
        h = np.clip(h, 0.0, 1.0)
        diff = TARGET - h.sum()
        if abs(diff) < 1e-13:
            break
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if not np.any(mask):
            break
        h[mask] += diff / mask.sum()
    h = np.clip(h, 0.0, 1.0)
    # final exact fix on any non-saturated entries
    diff = TARGET - h.sum()
    if abs(diff) > 1e-13:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _correlate_full(h, g):
    """Full cross-correlation, matching np.correlate(h, g, mode='full')."""
    return np.correlate(h, g, mode='full')


def _analytic_max_grad(h):
    """Exact sub-gradient of max_lag corr(h,1-h) w.r.t. h."""
    one_h = 1.0 - h
    corr = _correlate_full(h, one_h)
    k = int(np.argmax(corr))
    offset = k - (N - 1)
    d = np.zeros(N)
    # term 1: h[i] * (1-h[i-offset])  ->  d/dh[i] = (1-h[i-offset])
    idx1 = np.arange(N) - offset
    m1 = (idx1 >= 0) & (idx1 < N)
    d[m1] += (1.0 - h[idx1[m1]])
    # term 2: h[i+offset] * (1-h[i])  ->  d/dh[i] = -h[i+offset]
    idx2 = np.arange(N) + offset
    m2 = (idx2 >= 0) & (idx2 < N)
    d[m2] += -h[idx2[m2]]
    return d


def _smooth_max(h, beta):
    """Smooth (log-sum-exp) approximation of max(corr)."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    shifted = beta * (corr - m)
    return m + np.log(np.sum(np.exp(shifted))) / beta


def _smooth_max_grad(h, beta):
    """Analytic gradient of the smooth max."""
    corr = _correlate_full(h, 1.0 - h)
    m = np.max(corr)
    shifted = beta * (corr - m)
    w = np.exp(shifted)
    w /= w.sum()
    d = np.zeros(N)
    for k in range(2 * N - 1):
        if w[k] <= 1e-12:
            continue
        offset = k - (N - 1)
        idx1 = np.arange(N) - offset
        m1 = (idx1 >= 0) & (idx1 < N)
        d[m1] += w[k] * (1.0 - h[idx1[m1]])
        idx2 = np.arange(N) + offset
        m2 = (idx2 >= 0) & (idx2 < N)
        d[m2] += w[k] * (-h[idx2[m2]])
    return d


def _gd_exact(h0, max_iter=4000, lr0=0.05, deadline=None):
    """Projected gradient descent using the exact max sub-gradient."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    lr = lr0
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        d = _analytic_max_grad(h)
        # normalize step magnitude to control movement
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            break
        h_new = h - lr * (d / nrm)
        h_new = _project(h_new)
        val = _objective(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new
            lr = min(lr * 1.02, 0.2)
        else:
            lr *= 0.9
            if lr < 1e-5:
                break
    return best_h, best_val


def _gd_smooth(h0, max_iter=2000, lr0=0.02, deadline=None):
    """Projected gradient descent using the smooth max (escapes plateaus)."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    lr = lr0
    beta = 8.0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        d = _smooth_max_grad(h, beta)
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            break
        h_new = h - lr * (d / nrm)
        h_new = _project(h_new)
        val = _objective(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new
        else:
            lr *= 0.95
            beta *= 1.05  # sharpen toward exact max as we converge
            if lr < 1e-6:
                break
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)

    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)  # seed = 0.5

    rng = np.random.RandomState(100)
    starts = [best_h.copy()]

    # random perturbations of varying scale
    for scale in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for _ in range(2):
            hp = 0.5 + rng.randn(N) * scale
            starts.append(_project(hp))

    # sine waves
    x = np.arange(N)
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.2, 0.4, 0.45]:
            hs = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            starts.append(_project(hs))

    # alternating two-level
    for a in [0.2, 0.3, 0.4, 0.45]:
        ha = np.where(x % 2 == 0, a, 1.0 - a)
        starts.append(_project(ha))

    # block patterns
    for bs in [10, 20, 25]:
        hb = np.zeros(N)
        hb[::2 * bs][:bs] = 1.0
        for i in range(0, N, 2 * bs):
            hb[i:i + bs] = 1.0
        starts.append(_project(hb))

    # linear ramp
    starts.append(_project(np.linspace(0.0, 1.0, N)))

    # refine each start: exact GD then smooth GD polish
    for h0 in starts:
        if time.time() > deadline - 3:
            break
        h1, v1 = _gd_exact(h0, max_iter=4000, lr0=0.05, deadline=deadline)
        if v1 < best_val:
            best_val = v1
            best_h = h1.copy()
        # smooth polish from the best-so-far and from this exact-GD result
        h2, v2 = _gd_smooth(h1, max_iter=1500, lr0=0.02, deadline=deadline)
        if v2 < best_val:
            best_val = v2
            best_h = h2.copy()

    # final global polish on the best point
    if time.time() < deadline - 1:
        h3, v3 = _gd_smooth(best_h, max_iter=1500, lr0=0.01, deadline=deadline)
        if v3 < best_val:
            best_val = v3
            best_h = h3.copy()

    best_h = _project(best_h)
    return best_h
