"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time

N = 600

# EVOLVE-BLOCK-START
def _autoconv_peak(x):
    """Compute the peak of the autoconvolution of x (length N)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    c = np.fft.irfft(X * X, size)[:2 * n - 1]
    return np.max(c)

def _score(x):
    """The normalized maximum of autoconvolution."""
    h = 1.0 / (2 * N)
    integral = h * np.sum(x)
    if integral <= 0:
        return np.inf
    peak = _autoconv_peak(x) * h
    return peak / (integral * integral)

def _softplus_project(z, temperature=0.1):
    """Project z to nonnegative via softplus-like transformation."""
    # Use a smooth approximation: x = max(0, z) + temperature * log(1 + exp(-z/temperature))
    # This ensures non-negativity while maintaining differentiability
    z = np.asarray(z)
    x = np.maximum(0, z) + temperature * np.log1p(np.exp(-z / temperature))
    return x

def _freq_to_time(freq_coeffs):
    """Convert frequency domain coefficients to time domain step heights.
    
    freq_coeffs: complex array of length N (full DFT)
    Returns: nonnegative real array of length N
    """
    # Inverse DFT
    x = np.fft.ifft(freq_coeffs).real
    # Project to nonnegative using softplus
    x = _softplus_project(x)
    return x

def _time_to_freq(x):
    """Convert time domain to frequency domain."""
    return np.fft.fft(x)

def _compute_autoconv_from_freq(freq_coeffs):
    """Compute autoconvolution peak directly from frequency coefficients.
    
    For a signal x with DFT X, the autoconvolution c = x * x has DFT C = X^2.
    The time-domain autoconvolution is ifft(X^2).
    """
    X_sq = freq_coeffs * freq_coeffs
    c = np.fft.ifft(X_sq).real
    # For linear convolution of length N, we need to handle the wrap-around
    # The autoconvolution of length N signal has length 2N-1
    # ifft gives circular convolution, so we need to unwrap
    # Actually, for the peak, we can just compute the full linear convolution
    c_linear = np.fft.irfft(np.fft.rfft(np.fft.ifft(freq_coeffs).real, 2*N-1)**2, 2*N-1)
    return np.max(c_linear)

def _spectral_coordinate_descent(freq_coeffs, max_iters=5000, deadline=None):
    """Coordinate descent on frequency domain coefficients.
    
    For each frequency bin, adjust the magnitude and phase to minimize the peak autoconvolution.
    """
    best_freq = freq_coeffs.copy()
    best_score = _score(_freq_to_time(best_freq))
    
    n_freq = len(freq_coeffs)
    # Only optimize real and imaginary parts for positive frequencies
    # DC (index 0) and Nyquist (index N//2 if even) are real
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Iterate over frequency bins
        for k in range(n_freq):
            # For each bin, try adjusting magnitude and phase
            old_re = freq_coeffs[k].real
            old_im = freq_coeffs[k].imag
            
            # Try several magnitude and phase adjustments
            best_local = (old_re, old_im)
            best_local_score = best_score
            
            # Magnitude scaling factors
            mag_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            # Phase adjustments (in radians)
            phase_adj = [-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4]
            
            current_mag = np.sqrt(old_re**2 + old_im**2)
            current_phase = np.angle(freq_coeffs[k])
            
            for mf in mag_factors:
                for pa in phase_adj:
                    new_mag = current_mag * mf
                    new_phase = current_phase + pa
                    new_re = new_mag * np.cos(new_phase)
                    new_im = new_mag * np.sin(new_phase)
                    
                    # Ensure conjugate symmetry for real signal
                    freq_coeffs[k] = complex(new_re, new_im)
                    # Update conjugate bin
                    k_conj = (-k) % n_freq
                    if k_conj != k:
                        freq_coeffs[k_conj] = complex(new_re, -new_im)
                    
                    x = _freq_to_time(freq_coeffs)
                    sc = _score(x)
                    
                    if sc < best_local_score:
                        best_local_score = sc
                        best_local = (new_re, new_im)
                        improved = True
            
            # Restore original if no improvement
            if not improved:
                freq_coeffs[k] = complex(old_re, old_im)
                k_conj = (-k) % n_freq
                if k_conj != k:
                    freq_coeffs[k_conj] = complex(old_re, -old_im)
            else:
                # Apply best local improvement
                freq_coeffs[k] = complex(best_local[0], best_local[1])
                k_conj = (-k) % n_freq
                if k_conj != k:
                    freq_coeffs[k_conj] = complex(best_local[0], -best_local[1])
                
                if best_local_score < best_score:
                    best_score = best_local_score
                    best_freq = freq_coeffs.copy()
        
        if not improved:
            break
    
    return best_freq, best_score

def _parametric_search(deadline):
    """Search over parametric families in frequency domain.
    
    Try cosine-based, Gaussian-based, and plateau-based spectral profiles.
    """
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Cosine basis: x[n] = 1 + a * cos(2*pi*f*n/N)
    for a in [0.3, 0.5, 0.7, 0.9]:
        for f in [1, 2, 3, 5, 10, 20, 50, 100]:
            n = np.arange(N)
            x = 1.0 + a * np.cos(2 * np.pi * f * n / N)
            x = np.maximum(0, x)
            if np.sum(x) > 0:
                sc = _score(x)
                if sc < best_score:
                    best_score = sc
                    best_x = x.copy()
    
    # Gaussian profile in time domain
    for center in [N//4, N//2, 3*N//4]:
        for width in [N//10, N//5, N//3, N//2]:
            n = np.arange(N)
            x = np.exp(-((n - center)**2) / (2 * width**2))
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
    
    # Plateau profile
    for plateau_frac in [0.2, 0.3, 0.5, 0.7]:
        plateau_len = int(plateau_frac * N)
        start = (N - plateau_len) // 2
        x = np.zeros(N)
        x[start:start+plateau_len] = 1.0
        # Smooth edges
        edge = max(1, N // 20)
        for i in range(edge):
            t = i / edge
            x[start - edge + i] = t if start - edge + i >= 0 else 0
            x[start + plateau_len + i] = 1 - t if start + plateau_len + i < N else 0
        if np.sum(x) > 0:
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
    
    return best_x, best_score

def _local_refinement(x, deadline):
    """Local refinement using coordinate descent in time domain."""
    best_x = x.copy()
    best_score = _score(best_x)
    
    for coord_pass in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = x[i]
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                x[i] = old_val * factor
                s = np.sum(x)
                if s > 0:
                    sc = _score(x)
                    if sc < best_score:
                        best_score = sc
                        best_x = x.copy()
                        improved = True
                    else:
                        x[i] = old_val
                else:
                    x[i] = old_val
        
        if not improved:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Phase 1: Parametric search for good initial point
    x_init, score_init = _parametric_search(deadline)
    
    # Phase 2: Convert to frequency domain and do spectral coordinate descent
    freq = _time_to_freq(x_init)
    freq_best, score_freq = _spectral_coordinate_descent(freq, max_iters=2000, deadline=deadline)
    
    # Phase 3: Convert back to time domain and do local refinement
    x_refined = _freq_to_time(freq_best)
    x_final, score_final = _local_refinement(x_refined, deadline)
    
    # Choose the best result
    candidates = [
        (x_init, score_init),
        (x_refined, _score(x_refined)),
        (x_final, score_final),
    ]
    
    best_x = candidates[0][0]
    best_score = candidates[0][1]
    for x_c, sc_c in candidates:
        if sc_c < best_score:
            best_score = sc_c
            best_x = x_c
    
    return list(best_x)
# EVOLVE-BLOCK-END
