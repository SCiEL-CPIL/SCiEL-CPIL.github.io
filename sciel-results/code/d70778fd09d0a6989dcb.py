"""Optimized construction for the signed autocorrelation C3-prime task."""
import time
import numpy as np

N = 400


def _autoconv(x):
    """Full linear autoconvolution of x via FFT (zero-padded)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(x, size)
    c = np.fft.irfft(F * np.conj(F), size)
    return c[:2 * n - 1]


def _score(x):
    """C3' = max|autoconv| / (integral)^2. Integral = sum(x)."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))


def _local_search(x, deadline, iters_per_step=20000):
    """Coordinate / sign-flip local search minimizing _score."""
    x = x.copy()
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(12345)
    i = 0
    while time.time() < deadline:
        # try flipping a random subset of signs (block moves) to escape local minima
        block = rng.integers(1, max(2, n // 4))
        idx = rng.choice(n, size=block, replace=False)
        cand = x.copy()
        cand[idx] = -cand[idx]
        sc = _score(cand)
        if sc < best:
            best = sc
            x = cand
        # single sign flips
        for _ in range(iters_per_step):
            j = int(rng.integers(n))
            cand = x.copy()
            cand[j] = -cand[j]
            sc = _score(cand)
            if sc < best:
                best = sc
                x = cand
        i += 1
    return x, best


def _structured_seed(rng):
    """Generate a structured signed seed: alternating / low-correlation pattern."""
    # Start from a Barker-like / random ±1, then scale.
    x = rng.choice([-1.0, 1.0], size=N)
    # Add a small DC component so the integral is non-zero (avoid singular denominator)
    x += 0.1
    return x


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 110.0  # safety margin before the 150s kill
    rng = np.random.default_rng(2024)

    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Try several structured and random seeds, then local-search the best.
    seeds = []
    seeds.append(np.ones(N, dtype=np.float64))
    for _ in range(4):
        seeds.append(_structured_seed(rng))
    # A few pure random ±1 with DC offset
    for _ in range(3):
        x = rng.choice([-1.0, 1.0], size=N) + 0.05
        seeds.append(x)

    for s in seeds:
        if time.time() >= deadline:
            break
        x, sc = _local_search(s, deadline)
        if sc < best_score:
            best_score = sc
            best_x = x

    if time.time() >= deadline:
        return best_x
    return best_x
