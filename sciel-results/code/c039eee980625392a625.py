import time
import numpy as np

N = 50
_TIME_LIMIT = 125.0  # safety margin under the 150 s kill
_T0 = time.time()

def _elapsed():
    return time.time() - _T0

def _autoconv_peak(h):
    # linear autoconvolution, exact for nonnegative h (peak = max)
    n = len(h)
    m = 2 * n - 1
    f = np.fft.rfft(h, m)
    ac = np.fft.irfft(f * f, m)
    return float(np.max(ac))

def _ratio(h):
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0 or l2sq <= 0:
        return 0.0
    peak = _autoconv_peak(h)
    if peak <= 0:
        return 0.0
    return l2sq / (l1 * peak)

def _objective(h, beta):
    # maximize 2*log(||h||_2) - log(||h||_1) - log(softmax_peak(beta))
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0 or l2sq <= 0:
        return -1e18
    n = len(h)
    m = 2 * n - 1
    f = np.fft.rfft(h, m)
    ac = np.fft.irfft(f * f, m)
    ac = np.maximum(ac, 0.0)
    if beta < 1e-6:
        peak = float(np.max(ac))
    else:
        amax = float(np.max(ac))
        # softmax over ac scaled by beta -> exp(beta*(ac-amax))
        e = np.exp(beta * (ac - amax))
        peak = amax + float(np.log(np.sum(e) + 1e-300))
    if peak <= 0:
        return -1e18
    return 2.0 * 0.5 * np.log(l2sq) - np.log(l1) - np.log(peak)

def _project(h):
    # clip to nonnegative; if all zero, return ones
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        return np.ones(N)
    return h

def _lbfgsb(h, beta, iters, step):
    # simple projected gradient with backtracking on the log-objective
    h = np.asarray(h, dtype=float).copy()
    best_h = h.copy()
    best_obj = _objective(h, beta)
    for it in range(iters):
        if _elapsed() > _TIME_LIMIT:
            break
        # numerical gradient of the log-objective (central diff)
        g = np.zeros(N)
        for i in range(N):
            hp = h.copy(); hp[i] += step
            hm = h.copy(); hm[i] = max(hm[i] - step, 0.0)
            g[i] = (_objective(hp, beta) - _objective(hm, beta)) / (2 * step)
        # projected gradient ascent
        h2 = _project(h + 0.5 * g)
        # backtracking on the objective
        lr = 0.5
        for _ in range(30):
            cand = _project(h + lr * g)
            cobj = _objective(cand, beta)
            if cobj >= best_obj - 1e-12 or cobj > _objective(h, beta):
                h = cand
                break
            lr *= 0.5
        else:
            break
        cobj = _objective(h, beta)
        if cobj > best_obj:
            best_obj = cobj
            best_h = h.copy()
    return best_h

def construct():
    deadline = _TIME_LIMIT
    rng = np.random.default_rng(20240501)

    # Start from a variety of seeds and run projected gradient on the log-ratio.
    best_h = np.ones(N)
    best_r = _ratio(best_h)  # == 1.0

    def consider(cand):
        nonlocal best_h, best_r
        cand = _project(cand)
        r = _ratio(cand)
        if r > best_r + 1e-12:
            best_r = r
            best_h = cand.copy()

    # Candidate initial points
    inits = []
    inits.append(np.ones(N))
    inits.append(np.ones(N) * 0.5)
    # concentrated (spiky) starts
    for k in (1, 2, 3, 5):
        v = np.zeros(N); v[:k] = 1.0
        inits.append(v)
    # random nonnegative
    for _ in range(6):
        inits.append(np.abs(rng.standard_normal(N)) + 0.1)
    # cosine / ramp shapes
    x = np.linspace(0, 1, N)
    inits.append(np.abs(np.cos(np.pi * x)) + 0.1)
    inits.append(1.0 - x)
    inits.append(x)
    # a few low-frequency smooth bumps
    for _ in range(4):
        inits.append(np.maximum(0.0, np.sin(2 * np.pi * (x * rng.uniform(1, 4) + rng.uniform(0, 1)))) + 0.05)

    for v in inits:
        if _elapsed() > deadline:
            break
        consider(v)
        # gradient ascent from this start with a smooth surrogate
        h = _lbfgsb(v, beta=8.0, iters=40, step=0.05)
        consider(h)

    # Polish the best with the exact (nonsmooth) peak using smaller steps
    if _elapsed() < deadline - 5:
        h = _lbfgsb(best_h, beta=0.0, iters=60, step=0.02)
        consider(h)

    # Final exact ratio of the best found
    final = _project(best_h)
    _ratio(final)  # validate
    return [float(x) for x in final]

if __name__ == "__main__":
    out = construct()
    print("ratio:", _ratio(np.asarray(out)))
