"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(x):
    n = len(x)
    size = 2 * n
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * fx, size)
    return ac[:2 * n - 1]


def _score(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    return float(np.max(np.abs(ac)) / (s * s))


def construct():
    start = time.time()
    deadline = start + 140.0

    best_x = np.ones(N)
    best_val = _score(best_x)

    rng = np.random.default_rng(12345)

    # Try random ±1 and random real sequences
    while time.time() < deadline:
        # ±1 sequence
        x = rng.choice([-1.0, 1.0], size=N)
        v = _score(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()

        # Random real sequence with positive mean
        x = 1.0 + rng.normal(0, 0.5, N)
        v = _score(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
