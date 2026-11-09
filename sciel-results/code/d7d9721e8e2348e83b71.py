"""Erdos minimum overlap: spectral (minimax) optimization of step heights."""
import numpy as np
import time

N = 100
STEP = 2.0 / N          # each step width on [0,2]
TOTAL = N / 2.0         # required sum of heights = 50


def _overlap_spectrum(h):
    """Return the vector of overlaps O_k for every cyclic shift k.

    O_k = STEP * sum_i h_i * (1 - h_{i+k}) = 1 - (1/N) * C_h(k),
    where C_h(k) = sum_i h_i * h_{i+k} is the (cyclic) autocorrelation.
    Uses the FFT for a full-spectrum O( N log N ) evaluation.
    """
    H = np.fft.rfft(h)
    power = np.abs(H) ** 2                 # = DFT of the autocorrelation
    C = np.fft.irfft(power, n=N)           # C[k] = sum_i h_i * h_{i+k}
    return 1.0 - C / N


def _score(h):
    """Maximum overlap over all shifts (the quantity to minimize)."""
    return float(np.max(_overlap_spectrum(h)))


def _flat_autocorrelation_binary(N):
    """Find a binary b in {0,1}^N (50 ones) whose autocorrelation is as flat
    as possible at nonzero shifts (Barker / Lempel-Golomb style).

    Flat autocorrelation => flat overlap spectrum => low maximum overlap.
    Uses iterative thresholding on the non-DC Fourier magnitudes.
    """
    rng = np.random.default_rng(20240517)
    t0 = time.time()
    best = None
    best_metric = float('inf')

    def _autocorr(b):
        H = np.fft.rfft(b.astype(np.float64))
        return np.fft.irfft(np.abs(H) ** 2, n=N)

    def _metric(b):
        ac = _autocorr(b)
        # penalty on nonzero-shift spread; keep DC (ac[0]) unconstrained
        off = np.concatenate([ac[1:], ac[-1:][::-1]])
        return off.max() - off.min() + 1e-6 * abs(off.mean() - (N * 0.5))

    def _iterative_threshold(b, iters, T):
        c = b.copy().astype(np.float64)
        for _ in range(iters):
            if time.time() - t0 > 40.0:
                break
            H = np.fft.rfft(c)
            mag = np.abs(H)
            if mag[0] > 1e-12:
                mag[0] = 0.0
            c = np.fft.irfft(mag, n=N)
            c = (c > T).astype(np.float64)
            if c.sum() != TOTAL:
                order = np.argsort(-c)
                c = np.zeros(N)
                c[order[:TOTAL]] = 1.0
            else:
                c = c.astype(np.float64)
        return c

    for start in range(8):
        if time.time() - t0 > 45.0:
            break
        if start == 0:
            b = np.zeros(N)
            b[:TOTAL] = 1.0
        else:
            b = rng.random(N)
            b = (b > np.median(b)).astype(np.float64)
            if b.sum() != TOTAL:
                order = np.argsort(-b)
                b = np.zeros(N)
                b[order[:TOTAL]] = 1.0
        for iters in (80, 220):
            b2 = _iterative_threshold(b, iters, 0.5)
            m = _metric(b2)
            if m < best_metric:
                best_metric = m
                best = b2.copy()
            if time.time() - t0 > 50.0:
                break

    if best is None:
        best = np.zeros(N); best[:TOTAL] = 1.0
    # exact-sum / bound sanity
    if best.sum() != TOTAL:
        order = np.argsort(-best)
        best = np.zeros(N); best[order[:TOTAL]] = 1.0
    return best.astype(np.float64)


def _spectral_minimax(h0, deadline):
    """Minimax optimization in the frequency domain.

    Represent h via its rFFT; iterate: project current autocorrelation's
    overlap spectrum O_k toward flat (shrink peaks / lift troughs), lift back
    to the time domain, then clip to [0,1] and fix the sum. This drives the
    overlap spectrum flat, directly attacking the maximum.
    """
    h = h0.astype(np.float64).copy()
    best = h.copy()
    best_score = _score(h)
    t0 = time.time()

    def _fix(h):
        h = np.clip(h, 0.0, 1.0)
        d = TOTAL - h.sum()
        # distribute the residual on the least-affected (mid) elements
        if abs(d) > 1e-12:
            order = np.argsort(-np.abs(h - 0.5))
            step = d / N
            h = h + step
            h = np.clip(h, 0.0, 1.0)
            # final tiny correction on a free element
            d2 = TOTAL - h.sum()
            if abs(d2) > 1e-12:
                free = np.where((h > 1e-9) & (h < 1.0 - 1e-9))[0]
                if len(free):
                    h[free[0]] = h[free[0]] + d2
                    h = np.clip(h, 0.0, 1.0)
        return h

    alpha = 0.35
    for it in range(4000):
        if time.time() > deadline:
            break
        O = _overlap_spectrum(h)
        # move O toward its mean (=0.5), shrinking peaks and lifting troughs
        O_new = O + alpha * (O.mean() - O)
        # O_k = 1 - C_k/N  =>  C_k = N*(1 - O_k)
        C = N * (1.0 - O_new)
        C = np.fft.rfft(C)
        C = np.real(C)
        h = np.real(np.fft.irfft(C, n=N))
        h = _fix(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best = h.copy()
    return best, best_score


def construct():
    start = time.time()
    deadline = start + 135.0   # hard safety margin (evaluator kills at 150s)

    # 1) Strong structured seed: flat-autocorrelation binary -> h in [0.25,0.75]
    b = _flat_autocorrelation_binary(N)
    h0 = 0.25 + 0.5 * b
    h0 = _fix_sum(h0)
    best, best_score = h0, _score(h0)

    # 2) Spectral minimax refinement from the structured seed
    h_ref, s_ref = _spectral_minimax(h0, min(deadline - 5.0, start + 110.0))
    if s_ref < best_score:
        best, best_score = h_ref, s_ref

    # 3) Spectral minimax from the uniform seed as a backup (cheap, robust)
    if time.time() < deadline - 15.0:
        h_u, s_u = _spectral_minimax(np.full(N, 0.5), deadline - 3.0)
        if s_u < best_score:
            best, best_score = h_u, s_u

    best = _fix_sum(best)
    return best


def _fix_sum(h):
    """Clip to [0,1] and make the sum exactly N/2."""
    h = np.clip(h, 0.0, 1.0)
    d = TOTAL - h.sum()
    if abs(d) > 1e-12:
        h = h + d / N
        h = np.clip(h, 0.0, 1.0)
        d2 = TOTAL - h.sum()
        if abs(d2) > 1e-12:
            free = np.where((h > 1e-9) & (h < 1.0 - 1e-9))[0]
            if len(free):
                h[free[0]] = h[free[0]] + d2
                h = np.clip(h, 0.0, 1.0)
    return h
