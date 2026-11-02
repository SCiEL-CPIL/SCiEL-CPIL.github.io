"""Signed autocorrelation maximum absolute-value variant - optimal constant sequence."""
import numpy as np
import time

N = 400
DEADLINE = 120.0  # seconds


def _autoconv_max(x):
    """Compute the maximum absolute value of the linear autoconvolution of x."""
    # Use FFT for efficient linear convolution
    # Pad to 2*N to avoid circular wrap-around
    f = np.fft.fft(x, n=2 * N)
    conv = np.fft.ifft(f * np.conj(f)).real[:N]
    return np.max(np.abs(conv))


def _score(x):
    """Score: maximum absolute autoconvolution divided by squared integral (sum)."""
    autoconv = _autoconv_max(x)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return autoconv / (integral * integral)


def construct():
    start_time = time.time()
    
    # The theoretical minimum for the score max(|c_k|) / (sum x)^2 is 1/N.
    # This is achieved by the constant sequence x = [1, 1, ..., 1].
    # Any deviation from constant increases sum(x^2) relative to sum(x)^2,
    # and potentially increases sidelobes, leading to a higher score.
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Verify with a few other simple patterns just in case, though constant is optimal.
    # 1. Alternating
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    s_alt = _score(x_alt)
    if s_alt < best_score:
        best_x = x_alt
        best_score = s_alt

    # 2. Random binary (0, 1)
    x_rand = np.random.randint(0, 2, N).astype(float)
    s_rand = _score(x_rand)
    if s_rand < best_score:
        best_x = x_rand
        best_score = s_rand

    # 3. Random continuous
    x_cont = np.random.randn(N)
    s_cont = _score(x_cont)
    if s_cont < best_score:
        best_x = x_cont
        best_score = s_cont

    # 4. Step function (half 1, half 0)
    x_step = np.zeros(N)
    x_step[:N//2] = 1
    s_step = _score(x_step)
    if s_step < best_score:
        best_x = x_step
        best_score = s_step

    return best_x
