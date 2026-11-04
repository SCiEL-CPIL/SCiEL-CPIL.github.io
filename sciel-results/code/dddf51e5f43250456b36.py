"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autoconv_peak(f):
    """Compute the maximum absolute value of the discrete autoconvolution."""
    # f is array of length N with values ±1
    # Autoconvolution: (f * f)[k] = sum_i f[i] * f[i+k] for valid k
    # Using FFT for efficiency
    # Pad to avoid circular convolution effects
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    autoconv = np.fft.ifft(F * F).real
    # The linear autoconvolution is in the first N elements (and symmetric)
    # Take max absolute value over the full autoconvolution
    return np.max(np.abs(autoconv))


def _try_flip(f, idx):
    """Try flipping sign at index idx, return new peak if improved."""
    old_peak = _compute_autoconv_peak(f)
    f[idx] *= -1
    new_peak = _compute_autoconv_peak(f)
    if new_peak < old_peak:
        return True, new_peak
    else:
        f[idx] *= -1  # revert
        return False, old_peak


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Initialize with random signs
    np.random.seed(42)
    f = np.random.choice([-1.0, 1.0], size=N)
    
    current_peak = _compute_autoconv_peak(f)
    best_f = f.copy()
    best_peak = current_peak
    
    # Local search: try flipping each sign
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            f[i] *= -1
            new_peak = _compute_autoconv_peak(f)
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
            else:
                f[i] *= -1  # revert
        if time.time() > deadline:
            break
    
    # Second pass: more thorough search if time permits
    if time.time() < deadline - 10.0:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                f[i] *= -1
                new_peak = _compute_autoconv_peak(f)
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_f = f.copy()
                else:
                    f[i] *= -1  # revert
            if time.time() > deadline:
                break
    
    return best_f
# EVOLVE-BLOCK-END
