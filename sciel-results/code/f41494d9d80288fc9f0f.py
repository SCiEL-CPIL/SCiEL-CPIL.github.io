"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(x):
    # linear autocorrelation via FFT
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    P = F * np.conj(F)
    ac = np.fft.irfft(P, size)[:2 * n - 1]
    return ac


def _metric(x):
    ac = _autocorr(x)
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    # max absolute sidelobe (exclude lag 0)
    side = np.concatenate([ac[1:], ac[:0]])  # ac[1:] are lags 1..n-1 (positive side)
    # also negative lags: ac[n-1+? ] -- for real, ac is symmetric; ac[1:] covers lags 1..n-1
    peak = np.max(np.abs(ac[1:]))
    return peak / (S * S)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    rng = np.random.default_rng(12345)

    # ---- Continuous optimization ----
    x = rng.standard_normal(N)
    # project to sum = 1
    x = x / np.sum(x) if abs(np.sum(x)) > 1e-9 else x + 1.0/N

    best_x = x.copy()
    best_val = _metric(best_x)

    # smooth max of sidelobes
    def smooth_max_side(x, beta):
        ac = _autocorr(x)
        side = np.abs(ac[1:])
        m = side.max()
        return m + np.log(np.sum(np.exp((side - m) * beta)) / len(side)) / beta

    lr = 0.02
    beta = 20.0
    # gradient of smooth max w.r.t. x is expensive analytically; use finite-diff-free:
    # Instead use subgradient via the argmax sidelobe and its gradient.
    # We'll do projected gradient with numerical gradient of the (non-smooth) max
    # using the dominant sidelobe index.

    step = 0
    while time.time() < deadline - 30:
        ac = _autocorr(x)
        side = np.abs(ac[1:])
        k = int(np.argmax(side))  # lag index (1-based)
        # gradient of ac[k] w.r.t. x: ac[k] = sum_i x[i]*x[i+k] (for k>=1)
        # d/dx[j] = x[j-k] (if j>=k) + x[j+k] (if j+k < N)
        grad = np.zeros(N)
        # ac[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
        # d/dx[j]: terms where j=i -> x[j+k]; where j=i+k -> x[j-k]
        for j in range(N):
            g = 0.0
            if j + k < N:
                g += x[j + k]
            if j - k >= 0:
                g += x[j - k]
            grad[j] = g
        # sign of ac[k]
        sgn = np.sign(ac[k + 1])  # ac index: ac[0]=lag0, ac[k]=lag k
        # we want to reduce |ac[k]| -> move x opposite to sgn*grad
        x = x - lr * sgn * grad
        # project to sum=1
        x = x + (1.0 - np.sum(x)) / N
        # normalize scale-invariance: keep sum=1 done
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        step += 1
        # adapt beta/lr
        if step % 50 == 0:
            lr *= 0.9
        if step > 20000:
            break

    # ---- Round to +/-1 and local search ----
    x1 = np.sign(best_x)
    # ensure nonzero sum
    if np.sum(x1) == 0:
        x1[0] = 1.0
    val1 = _metric(x1)
    if val1 < best_val:
        best_val = val1
        best_x = x1.copy()

    # greedy flip local search on +/-1
    xcur = best_x.copy()
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        ac = _autocorr(xcur)
        side = np.abs(ac[1:])
        k = int(np.argmax(side))
        # try flipping each position, keep best
        base = side[k]
        best_flip = -1
        best_side = base
        for j in range(N):
            xcur[j] = -xcur[j]
            # recompute only affected lags? full recompute is O(N log N); N=400 fine but N flips per iter
            # Instead compute change in ac[k] only
            xcur[j] = -xcur[j]
        # full recompute per candidate is heavy; do a smarter local: flip single bit, compute new ac[k]
        xcur = best_x.copy()
        ac0 = _autocorr(xcur)
        k = int(np.argmax(np.abs(ac0[1:])))
        for j in range(N):
            xtry = xcur.copy()
            xtry[j] = -xtry[j]
            actry = _autocorr(xtry)
            newk_val = np.abs(actry[k + 1])
            # also check global max didn't blow up
            newmax = np.max(np.abs(actry[1:]))
            oldmax = np.max(np.abs(ac0[1:]))
            if newmax < oldmax - 1e-9:
                xcur = xtry
                ac0 = actry
                k = int(np.argmax(np.abs(ac0[1:])))
                improved = True
                break
        if improved:
            valc = _metric(xcur)
            if valc < best_val:
                best_val = valc
                best_x = xcur.copy()

    # final: return best
    return best_x
# EVOLVE-BLOCK-END
