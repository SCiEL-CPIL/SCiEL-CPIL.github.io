import numpy as np
import time

N = 600
dx = 0.5 / N

def _autoconv_max(heights):
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    return np.max(conv)

def construct():
    return np.ones(N).tolist()
