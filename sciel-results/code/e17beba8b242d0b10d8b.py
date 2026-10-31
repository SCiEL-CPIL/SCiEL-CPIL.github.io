"""Erdos minimum overlap: minimize max_s sum_i h_i (1 - h_{i+s})."""
import time
import numpy as np

N = 100
_DEADLINE = 145.0          # wall-clock safety margin (evaluator kills at 150s)
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _max_overlap(h):
    """Return max_s sum_i h_i (1 - h_{i+s}) via circular cross-correlation (FFT)."""
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    if n == 0:
        return 0.0
    f = np.fft.rfft(h, n=n)
    g = np.fft.rfft(1.0 - h, n=n)
    c = np.fft.irfft(f * np.conj(g), n=n)
    return float(np.max(c))


def _project(h, T):
    """Project onto box [0,1] and sum N/2 (bisection on a shift)."""
    h = np.asarray(h, dtype=np.float64)
    target = 0.5 * h.size
    lo, hi = -float(h.max()), -float(h.min())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(h + mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    return np.clip(h + 0.5 * (lo + hi), 0.0, 1.0)


def _smooth_max(c, T):
    """Soft-max of a vector with temperature T (returns scalar)."""
    T = max(T, 1e-9)
    m = c.max()
    if m <= 0.0:
        return 0.0
    z = c / T
    z -= z.max()
    return float(m + T * np.log(np.sum(np.exp(z))))


def _pocs(h, iters, T0):
    """Alternating projections minimizing the smoothed max overlap."""
    h = _project(h, T0)
    cur = _max_overlap(h)
    best = h.copy()
    best_v = cur
    T = T0
    for it in range(iters):
        if _time_left() < 3.0:
            break
        c = np.fft.irfft(np.fft.rfft(h) * np.conj(np.fft.rfft(1.0 - h)), n=N)
        w = np.exp(c / T)
        w /= w.sum()
        h = _project(h - w, T)
        if (it % 4 == 0) or it == iters - 1:
            v = _max_overlap(h)
            if v < best_v - 1e-12:
                best_v = v
                best = h.copy()
            if it > 50 and it % 25 == 0:
                T *= 0.8
    return best, best_v


def _split_bregman(h, iters, lam):
    """Convex subproblem: min (1/2)||h-h0||^2 + lam*TV(h) s.t. box, sum=N/2.

    TV = sum |h_i - h_{i+1}| (circular). Solved by split-Bregman with
    dual/box/sum corrections; then projected.
    """
    h0 = np.asarray(h, dtype=np.float64)
    u = h0.copy()
    d = np.zeros(N)
    d[0] = 1e-6
    rho = 1.0
    target = 0.5 * N
    for it in range(iters):
        if _time_left() < 3.0:
            break
        # u-update: (I + rho A^T A) u = h0 + rho A^T d
        # A^T A u = u - 2 shift + shift^2 (circular)
        us = np.roll(u, 1)
        us2 = np.roll(u, 2)
        lhs = u + rho * (u - 2.0 * us + us2)
        rhs = h0 + rho * (d - 2.0 * np.roll(d, 1) + np.roll(d, 2))
        # solve via FFT (diagonal in Fourier)
        F = np.fft.rfft(lhs)
        R = np.fft.rfft(rhs)
        u = np.fft.irfft(np.fft.rfft(rhs) / np.fft.rfft(lhs), n=N)
        # d-update: soft-threshold of (A u + d/rho)
        Au = np.roll(u, 1) - u
        d = np.sign(Au + d / rho) * np.maximum(np.abs(Au + d / rho) - 1.0 / rho, 0.0)
        # bregman update
        d += rho * (u - np.roll(u, 1))
        # enforce box + sum via projection
        u = _project(u, 1.0)
    return u


def _polish(h, iters):
    """Coordinate descent: minimize max overlap, keep box + exact sum N/2."""
    h = _project(h, 1.0)
    best = h.copy()
    best_v = _max_overlap(h)
    target = 0.5 * N
    for it in range(iters):
        if _time_left() < 2.0:
            break
        improved = False
        c = np.fft.irfft(np.fft.rfft(h) * np.fft.rfft(1.0 - h).conj(), n=N)
        cur = float(c.max())
        for i in range(N):
            # try small moves in +/- directions
            for dstep in (0.02, 0.05, 0.1):
                for sgn in (1.0, -1.0):
                    hn = h.copy()
                    hn[i] = np.clip(h[i] + sgn * dstep, 0.0, 1.0)
                    if abs(hn.sum() - target) > 1e-9:
                        continue
                    vn = _max_overlap(hn)
                    if vn < cur - 1e-9:
                        h = hn
                        cur = vn
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # try a global re-balance nudge to keep sum exact
            break
        if cur < best_v - 1e-12:
            best_v = cur
            best = h.copy()
    return best, best_v


def construct():
    h = np.full(N, 0.5)
    best = h.copy()
    best_v = _max_overlap(best)

    # 1) POCs minimax from a few structured starts
    starts = [
        np.full(N, 0.5),
        np.tile([0.0, 1.0], N // 2),
        np.tile([0.0, 0.0, 1.0, 1.0], N // 4),
        0.5 + 0.5 * np.cos(2.0 * np.pi * 1 * np.arange(N) / N),
        0.5 + 0.5 * np.cos(2.0 * np.pi * 3 * np.arange(N) / N),
    ]
    for s in starts:
        if _time_left() < 6.0:
            break
        h0 = _project(s, 1.0)
        h1, v1 = _pocs(h0, 400, T0=0.25)
        if v1 < best_v - 1e-9:
            best, best_v = h1, v1

    # 2) split-Bregman TV pass to inject structure, then re-POCs
    if _time_left() > 8.0:
        hb = _split_bregman(best, 200, lam=0.5)
        h1, v1 = _pocs(_project(hb, 1.0), 300, T0=0.2)
        if v1 < best_v - 1e-9:
            best, best_v = h1, v1

    # 3) coordinate polish (exact box + sum)
    if _time_left() > 4.0:
        h1, v1 = _polish(best, 40)
        if v1 < best_v - 1e-9:
            best, best_v = h1, v1

    # final exact projection
    best = _project(best, 1.0)
    return best
