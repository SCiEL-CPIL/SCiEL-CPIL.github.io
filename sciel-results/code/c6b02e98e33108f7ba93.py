"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sequence of 400 signed step heights.
    
    Strategy: Use a windowed Zadoff-Chu sequence combined with spectral shaping
    to minimize the autoconvolution peak relative to sum(h)^2.
    
    The key insight from observations is that combining windowed Zadoff-Chu
    (near-zero autocorrelation in central region) with DCT spectral shaping
    can suppress the autoconvolution peak.
    """
    N = 400
    deadline = time.time() + 120  # 120s safety margin
    
    def compute_value(h):
        """Compute the objective value."""
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    def zadoff_chu(N, q):
        """Generate Zadoff-Chu sequence of length N with parameter q."""
        # Find valid q (coprime with N, 1 < q < N)
        for qq in range(1, N):
            if np.gcd(qq, N) == 1:
                q = qq
                break
        n = np.arange(N)
        phase = -np.pi * q * n * (n + 1) / N
        return np.exp(1j * phase)
    
    def windowed_zc(N, window_type='hann'):
        """Generate windowed Zadoff-Chu real sequence."""
        zc = zadoff_chu(N, 3)
        # Take real part and apply window
        if window_type == 'hann':
            w = np.hanning(N)
        elif window_type == 'hamming':
            w = np.hamming(N)
        elif window_type == 'blackman':
            w = np.blackman(N)
        else:
            w = np.ones(N)
        h = np.real(zc) * w
        # Ensure non-zero sum
        if abs(np.sum(h)) < 1e-10:
            h = h + 1.0
        return h
    
    # Start with windowed Zadoff-Chu
    h = windowed_zc(N, 'hann')
    best_h = h.copy()
    best_val = compute_value(h)
    
    # Try different windows and parameters
    for window_type in ['hann', 'hamming', 'blackman']:
        for q_offset in range(0, 10):
            if time.time() > deadline - 30:
                break
            N_local = N
            # Generate ZC with different q
            for qq in range(1, N_local):
                if np.gcd(qq, N_local) == 1:
                    n = np.arange(N_local)
                    phase = -np.pi * qq * n * (n + 1) / N_local
                    zc = np.exp(1j * phase)
                    if window_type == 'hann':
                        w = np.hanning(N_local)
                    elif window_type == 'hamming':
                        w = np.hamming(N_local)
                    else:
                        w = np.blackman(N_local)
                    h_candidate = np.real(zc) * w
                    if abs(np.sum(h_candidate)) > 1e-10:
                        val = compute_value(h_candidate)
                        if val < best_val:
                            best_val = val
                            best_h = h_candidate.copy()
                    break  # Only first valid q for speed
    
    # DCT spectral shaping: modify in frequency domain
    if time.time() < deadline - 30:
        for alpha in np.linspace(0.0, 1.0, 11):
            if time.time() > deadline - 30:
                break
            h_work = best_h.copy()
            # Apply spectral shaping via DCT
            dct_coeffs = np.fft.fft(h_work)
            # Attenuate high frequencies
            freqs = np.fft.fftfreq(N)
            mask = 1.0 - alpha * np.abs(freqs) * 2  # reduce high freq
            mask = np.clip(mask, 0.0, 1.0)
            dct_coeffs_shaped = dct_coeffs * mask
            h_shaped = np.real(np.fft.ifft(dct_coeffs_shaped))
            if abs(np.sum(h_shaped)) > 1e-10:
                val = compute_value(h_shaped)
                if val < best_val:
                    best_val = val
                    best_h = h_shaped.copy()
    
    # Local optimization: gradient-free search
    if time.time() < deadline - 10:
        # Try random perturbations
        rng = np.random.default_rng(42)
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            h_try = best_h + rng.normal(0, 0.01, N)
            if abs(np.sum(h_try)) > 1e-10:
                val = compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
    
    # Ensure validity
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
