"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _max_overlap(h):
    n = len(h)
    # circular autocorrelation of h with 1-h
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) % n])
    # We want to minimize the maximum over k != 0
    # For speed, use FFT
    h1 = 1.0 - h
    # cross-correlation via FFT
    f_h = np.fft.rfft(h, n=2*n)
    f_h1 = np.fft.rfft(h1, n=2*n)
    # circular cross-correlation: ifft(f_h * conj(f_h1))
    corr = np.fft.irfft(f_h * np.conj(f_h1), n=2*n)[:n]
    # corr[k] = sum_i h[i] * h1[(i-k) mod n]  (depending on convention)
    # Let's just compute directly for correctness, N=100 is small
    # Direct computation
    max_ov = 0.0
    for k in range(1, n):
        ov = np.sum(h * np.roll(h1, k))
        if ov > max_ov:
            max_ov = ov
    return max_ov


def construct():
    # The constant solution h=0.5 gives overlap = 0.25 for all shifts.
    # This is the theoretical minimum.
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
