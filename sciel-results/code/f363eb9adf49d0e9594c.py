import time
import numpy as np

N = 400


def _score(x):
    """
    Return (max_abs_autoconv, int_sq) for a signed step function x (length N, entries in {-1,1}).
    int_sq = (sum x)^2.  We minimize max_abs_autoconv / int_sq.
    """
    # Linear autocorrelation via FFT (full, not circular)
    n = len(x)
    # Pad to at least 2n-1, use next power of two for FFT speed
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:2 * n - 1]
    # ac[k] for k=0..2n-2 corresponds to sum_i x[i] x[i+k] for k>=0 and symmetric for negative lags
    # Actually, the standard linear autocorrelation (non-circular) is:
    #   c[k] = sum_{i} x[i] * x[i+k]  for k = 0, ..., n-1
    # and c[-k] = c[k]. The FFT-based full autoconvolution gives all 2n-1 lags.
    # The "autoconvolution" as used in C3' typically refers to the full set of lags, so we take max abs over all.
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    int_sq = s * s
    if int_sq == 0:
        return max_abs, 1.0  # avoid div by zero
    return max_abs, int_sq


def construct():
    t_start = time.time()
    deadline = t_start + 140  # 10s safety margin before 150s kill

    x = np.ones(N, dtype=np.int8)
    best_ratio = _score(x)[0] / _score(x)[1]
    best_x = x.copy()

    # Simple local search: random flips, accept if ratio improves
    # Use a schedule: try many flips, keep best
    rng = np.random.default_rng(42)

    # Precompute some structure for speed: we'll flip one index at a time
    # But recompute full FFT each time is O(N log N) ~ 400*9 ≈ 3600 ops, fast enough for ~50k iterations in 140s?
    # Let's be conservative.

    # Alternative: block-based search. Represent x as blocks of consecutive same-sign.
    # But start with simple per-bit flips first.

    # Let's try a smarter approach: start with a structured initial guess.
    # The optimal signed step function for minimizing max autocorrelation / (sum)^2 
    # is related to Barker sequences or low-autocorrelation binary sequences.
    # For N=400, we can try known good initial patterns.

    # Initialize with a pseudo-random balanced sequence
    x = rng.choice([-1, 1], size=N)
    max_abs, int_sq = _score(x)
    best_ratio = max_abs / int_sq if int_sq > 0 else 1e18
    best_x = x.copy()

    # Local search: try flipping each position, accept best single flip
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        current_ratio = best_ratio
        for i in range(N):
            x[i] = -x[i]
            ma, iq = _score(x)
            if iq > 0:
                r = ma / iq
            else:
                r = 1e18
            x[i] = -x[i]
            if r < current_ratio:
                current_ratio = r
                best_x[i] = -best_x[i]
                improved = True
        best_ratio = current_ratio

    # If time remains, try multi-flip local search
    if time.time() < deadline - 10:
        # Try flipping pairs
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            current_ratio = best_ratio
            for i in range(N):
                for j in range(i + 1, N):
                    x[i] = -x[i]
                    x[j] = -x[j]
                    ma, iq = _score(x)
                    if iq > 0:
                        r = ma / iq
                    else:
                        r = 1e18
                    x[i] = -x[i]
                    x[j] = -x[j]
                    if r < current_ratio:
                        current_ratio = r
                        best_x[i] = -best_x[i]
                        best_x[j] = -best_x[j]
                        improved = True
            best_ratio = current_ratio

    return best_x
