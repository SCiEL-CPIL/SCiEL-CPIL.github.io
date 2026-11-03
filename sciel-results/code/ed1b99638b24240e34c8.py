"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use Radon-Haar wavelet representation to exploit piecewise-constant structure.
2. Greedy coordinate descent on wavelet coefficients to minimize peak autoconvolution.
3. Local refinement in time domain using sensitivity-based peak reduction.
4. Final polish pass with tighter tolerances.

Key improvements:
- Wavelet basis provides sparsity and orthogonality, avoiding local minima
- Coordinate descent allows targeted adjustments to reduce specific peaks
- Combined with time-domain refinement for final polish
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
    return conv

def build_haar_basis(n):
    """Build Radon-Haar wavelet basis for length n.
    
    Returns a list of basis vectors, each of length n.
    The basis includes:
    - Level 0: constant vector (1/sqrt(n)) * ones(n)
    - Level k: 2^k wavelets, each supported on a dyadic interval
    """
    basis = []
    # Level 0: constant
    basis.append(np.ones(n) / np.sqrt(n))
    
    # Build Haar wavelets
    level = 0
    support = n
    while support >= 2:
        num_wavelets = n // support
        half = support // 2
        for idx in range(num_wavelets):
            start = idx * support
            w = np.zeros(n)
            w[start:start+half] = 1.0 / np.sqrt(support)
            w[start+half:start+support] = -1.0 / np.sqrt(support)
            basis.append(w)
        level += 1
        support //= 2
    
    return basis

def project_to_nonnegative(f):
    """Project f onto nonnegative orthant."""
    return np.maximum(f, 0.0)

def normalize_sum(f, target=N):
    """Normalize f to have sum = target."""
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros_like(f)
    return f * (target / s)

def wavelet_coordinate_descent(f_init, start_time, max_time, max_levels=8):
    """Greedy coordinate descent in Radon-Haar wavelet domain.
    
    For each wavelet coefficient, try small positive and negative updates
    and keep the one that reduces the peak autoconvolution.
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    
    # Build wavelet basis
    basis = build_haar_basis(N)
    num_coeffs = min(len(basis), 1 + (1 << max_levels) - 1)  # Limit levels
    basis = basis[:num_coeffs]
    
    # Compute initial wavelet coefficients
    coeff = np.array([np.dot(current_f, b) for b in basis])
    
    # Adaptive step size
    step_size = 0.1
    min_step = 1e-6
    
    # Precompute for efficiency
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        improved = False
        for k in range(1, num_coeffs):  # Skip constant term
            if time.time() - start_time > max_time - 2:
                break
            
            b = basis[k]
            c = coeff[k]
            
            # Try positive and negative updates
            for delta in [step_size, -step_size]:
                new_c = c + delta
                if new_c < 0:
                    continue  # Keep nonnegative in wavelet domain? Actually no, just adjust f
                
                # Update f
                f_new = current_f + delta * b
                f_new = project_to_nonnegative(f_new)
                f_new = normalize_sum(f_new)
                
                # Compute score
                conv = fft_autocorr(f_new)
                val = np.max(conv) / (N * N)
                
                if val < best_val:
                    best_val = val
                    best_f = f_new.copy()
                    current_f = f_new.copy()
                    coeff[k] = new_c
                    improved = True
                    step_size = min(step_size * 1.05, 0.5)
                    break
            else:
                # No improvement, try smaller step
                step_size *= 0.95
                if step_size < min_step:
                    step_size = min_step
        
        if not improved and step_size <= min_step:
            break
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.02, threshold_frac=0.98):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        # Compute sensitivity using FFT
        peak_ind = np.zeros(fft_size)
        peak_ind[peak_indices] = 1.0
        
        f_padded = np.zeros(fft_size)
        f_padded[:N] = current_h
        
        F = np.fft.rfft(f_padded)
        IP = np.fft.rfft(peak_ind)
        S = np.fft.irfft(F * np.conj(IP), n=fft_size)
        sensitivity = S[:N]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.95
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.005
    
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.999
        peak_indices = np.where(conv >= threshold)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])
        
        peak_ind = np.zeros(fft_size)
        peak_ind[peak_indices] = 1.0
        
        f_padded = np.zeros(fft_size)
        f_padded[:N] = current_h
        
        F = np.fft.rfft(f_padded)
        IP = np.fft.rfft(peak_ind)
        S = np.fft.irfft(F * np.conj(IP), n=fft_size)
        sensitivity = S[:N]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.01, 0.05)
        else:
            alpha *= 0.98
            if alpha < 1e-12:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try box function initialization
    box_widths = [0.1, 0.2, 0.3, 0.4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    for w in box_widths:
        f_box = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f_box[mask] = 1.0
        score_box = compute_score(f_box)
        if score_box < best_score:
            best_f = f_box
            best_score = score_box
    
    # Try a few structured initializations
    # Symmetric bumps
    for n_bumps in [2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        for spacing in [0.1, 0.15, 0.2]:
            f_init = np.zeros(N)
            for i in range(n_bumps):
                center = -0.2 + i * spacing
                if -0.25 <= center <= 0.25:
                    mask = np.abs(x - center) <= 0.05
                    f_init[mask] = 1.0
            if np.any(f_init > 0):
                f_init = normalize_sum(f_init)
                score = compute_score(f_init)
                if score < best_score:
                    best_f = f_init
                    best_score = score
    
    # Wavelet coordinate descent
    if time.time() - start_time < TIME_LIMIT - 30:
        wavelet_f, wavelet_score = wavelet_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 25, max_levels=6
        )
        if wavelet_score < best_score:
            best_f = wavelet_f
            best_score = wavelet_score
    
    # Local refinement: sensitivity-based peak reduction
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 1:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
