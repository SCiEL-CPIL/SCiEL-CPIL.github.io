"""Signed step heights minimizing max |autoconvolution| / (sum)^2 (C3-prime)."""
import time
import numpy as np

N = 400
_KILL = 150.0
_DEADLINE = _KILL - 25.0  # explicit safety margin before the evaluator kills us


def _autocorr_full(x):
    """Full (non-negative lag) autocorrelation C(k) = sum_i x_i x_{i-k}, k=0..n-1."""
    n = len(x)
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    X = np.fft.rfft(x, m)
    ac = np.fft.irfft(X * np.conj(X), m)
    return ac[:n]


def _objective(x):
    """Peak sidelobe absolute autoconvolution divided by squared sum (lower is better)."""
    s = float(x.sum())
    if abs(s) < 1e-9:
        return np.inf
    c = _autocorr_full(x)
    peak = float(np.max(np.abs(c[1:]))) if len(c) > 1 else 0.0
    return peak / (s * s)


def _rudin_shapiro(n):
    """Rudin-Shapiro sequence: provably low peak-to-mean autocorrelation."""
    a = np.array([1.0])
    b = np.array([1.0])
    while len(a) < n:
        a = np.concatenate([a, b])
        b = np.concatenate([a, -b])
    return a[:n].copy()


def _inits():
    """Diverse starting points; the constant seed is always present as a safe fallback."""
    out = [np.ones(N)]
    rs = _rudin_shapiro(N)
    out.append(rs)
    out.append(-rs)
    rng = np.random.default_rng(12345)
    for _ in range(8):
        out.append(rng.choice([-1.0, 1.0], size=N))
    return out


def _hill_climb(x, budget):
    """Sign-flip hill climbing with reversible kicks to escape local minima."""
    best = x.copy()
    best_obj = _objective(best)
    rng = np.random.default_rng(7)
    t0 = time.time()
    stall = 0
    while (time.time() - t0) < budget and time.time() < _DEADLINE:
        order = rng.permutation(N)
        any_improve = False
        for i in order:
            if time.time() > _DEADLINE:
                return best, best_obj
            best[i] *= -1.0
            o = _objective(best)
            if o < best_obj - 1e-15:
                best_obj = o
                any_improve = True
            else:
                best[i] *= -1.0
        if any_improve:
            stall = 0
        else:
            # reversible kick: perturb, keep only if it helps
            saved = best.copy()
            for _ in range(4):
                best[rng.integers(N)] *= -1.0
            o = _objective(best)
            if o < best_obj - 1e-15:
                best_obj = o
            else:
                best = saved
            stall += 1
            if stall > 8:
                break
    return best, best_obj


def construct():
    t_start = time.time()
    best = None
    best_obj = np.inf
    for c in _inits():
        if time.time() > _DEADLINE - 15:
            break
        o = _objective(c)
        if o < best_obj:
            best_obj = o
            best = c.copy()
    if best is None:
        return np.ones(N)
    remaining = _DEADLINE - 10 - (time.time() - t_start)
    if remaining > 2:
        best, best_obj = _hill_climb(best, remaining)
    return best
