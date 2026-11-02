"""Autoconvolution peak-sidelobe minimizer for the third autocorrelation inequality.

Metric (explicitly selected): the upper bound on C3 is taken as the peak
absolute value of the autoconvolution of the signed step function at
non-zero lags, i.e.  M(s) = max_{1 <= k < N} | (s * s)[k] |.
We place the absolute value per-lag and then take the maximum
(max_k |c[k]|), as opposed to |max_k c[k]|.  Minimizing M(s) lowers the
autoconvolution peak and hence the C3 upper bound (lower is better).

Strategy: steepest-descent greedy sign flips.  Each candidate flip at
position i changes the autoconvolution by an O(N) delta, so the new peak
is evaluated in O(N).  We repeatedly apply the single flip that most
reduces the peak until no improving flip remains, then random-restart from
low-discrepancy (Halton base-2) initializations, keeping the best.  A
soft-max (log-sum-exp) polish is used as a smooth surrogate to escape
flat regions.  A wall-clock deadline guard returns the best result found.
"""
import time
import numpy as np

N = 400
_L = 1024  # next power of two >= 2N, for FFT autoconvolution
_EPS = 1e-9


def _halton(n, base):
    """Base-`base` radical inverse (Halton) of integer n, in [0,1)."""
    f = 1.0
    r = 0.0
    i = n
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _init_halton(base, seed_shift=0):
    """Signed step heights from a Halton (low-discrepancy) sign pattern."""
    s = np.empty(N, dtype=np.float64)
    for i in range(N):
        h = _halton(i + 1 + seed_shift, base)
        s[i] = 1.0 if h < 0.5 else -1.0
    return s


def _autoconv(s):
    """Linear autoconvolution c[k] = sum_i s[i] s[i+k], k = 0..N-1."""
    S = np.fft.fft(s, _L)
    c = np.fft.ifft(S * S).real
    return c[:N]


def _peak(c):
    """Peak absolute autoconvolution at non-zero lags."""
    return float(np.max(np.abs(c[1:N])))


def _flip_delta(s, i):
    """Delta to the autoconvolution (lags 0..N-1) when flipping s[i]."""
    d = np.zeros(N, dtype=np.float64)
    si = s[i]
    # s[i+k] for k = 1..N-1-i  -> s[i+1 : N]
    if i + 1 < N:
        d[1:N - i] += -2.0 * si * s[i + 1:N]
    # s[i-k] for k = 1..i  -> s[0:i][::-1]
    if i > 0:
        d[1:i + 1] += -2.0 * si * s[0:i][::-1]
    return d


def _steepest(s, c, peak, deadline, max_iter=4000):
    """Steepest-descent greedy: apply the single best flip each iteration."""
    for _ in range(max_iter):
        if time.time() >= deadline:
            break
        best_i = -1
        best_peak = peak
        best_c = None
        for i in range(N):
            d = _flip_delta(s, i)
            nc = c + d
            npk = _peak(nc)
            if npk < best_peak - _EPS:
                best_peak = npk
                best_i = i
                best_c = nc
        if best_i < 0:
            break
        s[best_i] = -s[best_i]
        c = best_c
        peak = best_peak
    return s, c, peak


def _softmax_polish(s, c, peak, deadline, beta=8.0, max_iter=800):
    """Smooth surrogate: minimize log-sum-exp of |c[k]| (k>=1) by greedy flips."""
    def sm(c):
        a = np.abs(c[1:N]) * beta
        a = a - a.max()
        return float(np.log(np.sum(np.exp(a))) + a.max() / beta)
    cur = sm(c)
    for _ in range(max_iter):
        if time.time() >= deadline:
            break
        best_i = -1
        best_sm = cur
        best_c = None
        for i in range(N):
            d = _flip_delta(s, i)
            nc = c + d
            v = sm(nc)
            if v < best_sm - 1e-9:
                best_sm = v
                best_i = i
                best_c = nc
        if best_i < 0:
            break
        s[best_i] = -s[best_i]
        c = best_c
        cur = best_sm
        # track exact peak too
        if _peak(c) < peak:
            peak = _peak(c)
    return s, c, peak


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150 s wall-clock kill

    best_s = None
    best_peak = np.inf

    # A set of low-discrepancy / random initializations to try.
    inits = []
    for base in (2, 3, 5):
        inits.append(_init_halton(base))
    rng = np.random.default_rng(12345)
    for _ in range(3):
        inits.append(rng.choice([-1.0, 1.0], size=N))

    for s0 in inits:
        if time.time() >= deadline:
            break
        s = s0.copy()
        c = _autoconv(s)
        peak = _peak(c)
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
        # steepest-descent greedy
        s, c, peak = _steepest(s, c, peak, deadline)
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
        # soft-max polish (smooth surrogate) to escape flat regions
        s, c, peak = _softmax_polish(s, c, peak, deadline)
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()
        # re-run a short steepest pass after polish
        if time.time() < deadline - 2.0:
            s, c, peak = _steepest(s, c, peak, deadline, max_iter=2000)
            if peak < best_peak:
                best_peak = peak
                best_s = s.copy()

    if best_s is None:
        best_s = _init_halton(2)
    return best_s
