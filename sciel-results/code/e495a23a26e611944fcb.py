"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _van_der_corput_signs(n):
    """Generate n signs based on van der Corput sequence."""
    signs = np.zeros(n)
    for i in range(n):
        # Van der Corput: reverse binary digits
        v = 0.0
        frac = 0.5
        x = i
        while x > 0:
            if x % 2 == 1:
                v += frac
            x //= 2
            frac /= 2.0
        # Map to sign: +1 if v < 0.5, -1 otherwise
        signs[i] = 1.0 if v < 0.5 else -1.0
    return signs

def _compute_autoconv_peak(heights):
    """Compute the peak absolute value of autoconvolution at non-zero shifts."""
    n = len(heights)
    # Autoconvolution via FFT for efficiency
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.zeros(size)
    f[:n] = heights
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    # The autoconvolution at shift 0 is at index 0
    # Non-zero shifts are indices 1 to size-1
    # But we need to consider only valid shifts
    # For a function supported on [0, n), autoconv support is [0, 2n-1]
    # Peak at non-zero shifts
    peak = 0.0
    for k in range(1, size):
        if abs(autoconv[k]) > peak:
            peak = abs(autoconv[k])
    return peak

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate van der Corput signed step function
    heights = _van_der_corput_signs(N)
    
    # Compute initial peak
    current_peak = _compute_autoconv_peak(heights)
    
    # Local search: flip individual signs to reduce peak
    best_heights = heights.copy()
    best_peak = current_peak
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping sign i
            heights[i] = -heights[i]
            new_peak = _compute_autoconv_peak(heights)
            if new_peak < best_peak:
                best_peak = new_peak
                best_heights[i] = heights[i]
                improved = True
            else:
                heights[i] = -heights[i]  # revert
        if not improved:
            break
    
    return best_heights
# EVOLVE-BLOCK-END
