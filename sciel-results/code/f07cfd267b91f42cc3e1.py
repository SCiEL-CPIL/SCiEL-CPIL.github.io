"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Spectral coordinate descent in the frequency domain.
In frequency space, autoconvolution becomes pointwise |F|^2, so we can
shape the power spectrum to flatten the autocorrelation peak while
projecting back to non-negativity in time domain. This provides a
fundamentally different optimization path than time-domain gradient methods.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

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

def spectral_coordinate_descent(f0, start_time, max_time, n_iters=5000):
    """
    Spectral coordinate descent: alternate between shaping the power spectrum
    in frequency domain and projecting to non-negative time domain.
    
    Key idea: In frequency domain, autocorrelation peak is controlled by
    how evenly |F(omega)|^2 is distributed. We iteratively:
    1. Take FFT of current f
    2. Shape |F| to reduce concentration (flatten power spectrum)
    3. Take IFFT and project to non-negative
    4. Normalize
    """
    f = np.maximum(f0, 0.0).copy()
    s = np.sum(f)
    if s < 1e-15:
        f = np.ones(N) / N
    else:
        f = f / s
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    # Use a padded FFT for efficient convolution
    Nfft = 2 * N
    rng = np.random.RandomState(123)
    
    # Initial spectral shape: start with a smooth bump
    freqs = np.fft.fftfreq(Nfft, d=1.0/Nfft)
    
    for it in range(n_iters):
        if time.time() - start_time > max_time:
            break
        
        # FFT of current signal (zero-padded)
        F = np.fft.fft(f, Nfft)
        
        # Get power spectrum
        power = np.abs(F)**2
        
        # Shape the power spectrum to flatten autocorrelation
        # The autocorrelation is the IFFT of the power spectrum.
        # We want to reduce the peak of the autocorrelation.
        
        # Compute current autocorrelation in frequency domain
        # autocorr = ifft(power)
        # We want to modify power to reduce the max of ifft(power)
        
        # Strategy: apply a soft threshold to the power spectrum
        # High-power bins contribute more to autocorrelation peaks
        # We dampen high-power bins and boost low-power bins
        
        # Adaptive shaping: use the current autocorrelation peak location
        autocorr = np.fft.ifft(power).real
        peak_idx = np.argmax(autocorr)
        
        # Shape factor: reduce power where it contributes to the peak
        # Use a smooth shaping function
        power_max = np.max(power)
        power_min = np.min(power)
        
        if power_max > 1e-15:
            # Normalize power for shaping
            p_norm = power / power_max
            
            # Apply shaping: compress dynamic range
            # gamma < 1 compresses, gamma > 1 expands
            gamma = 0.7 + 0.3 * (it / min(n_iters, 100))  # start compressive, gradually relax
            
            # Shape: reduce high-power bins
            shaped_power = p_norm ** gamma
            
            # Add small floor to avoid zeroing out
            shaped_power = np.maximum(shaped_power, 0.01)
            
            # Renormalize to preserve total energy
            shaped_power = shaped_power / np.sum(shaped_power) * np.sum(power)
        else:
            shaped_power = power
        
        # Reconstruct F with shaped magnitude, keep original phase
        # But we can also adjust phase to help reduce autocorrelation peak
        new_mag = np.sqrt(shaped_power)
        
        # Keep phase from original F
        phase = np.angle(F)
        F_new = new_mag * np.exp(1j * phase)
        
        # IFFT back to time domain
        f_new = np.fft.ifft(F_new).real[:N]
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # Normalize
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            f_new = f.copy()
        else:
            f_new = f_new / s_new
        
        # Evaluate
        score_new = compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
        
        # Update f for next iteration
        f = f_new
    
    return best_f, best_score

def spectral_annealing(f0, start_time, max_time):
    """
    Simulated annealing in the spectral domain.
    Randomly perturb the phase of the FFT, project to non-negative time domain.
    This explores a different part of the solution space.
    """
    f = np.maximum(f0, 0.0).copy()
    s = np.sum(f)
    if s < 1e-15:
        f = np.ones(N) / N
    else:
        f = f / s
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    Nfft = 2 * N
    rng = np.random.RandomState(456)
    
    n_steps = 2000
    T_start = 0.1
    T_end = 0.001
    
    for it in range(n_steps):
        if time.time() - start_time > max_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / n_steps)
        
        # FFT
        F = np.fft.fft(f, Nfft)
        
        # Perturb phase randomly
        phase_perturb = rng.normal(0, T, Nfft)
        F_new = F * np.exp(1j * phase_perturb)
        
        # IFFT and project
        f_new = np.fft.ifft(F_new).real[:N]
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            continue
        f_new = f_new / s_new
        
        score_new = compute_score(f_new)
        
        # Accept if better or with probability exp(-delta/T)
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
        
        delta = score_new - compute_score(f)
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
            f = f_new
    
    return best_f, best_score

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Spectral coordinate descent from diverse initializations
    n_cd_restarts = 4
    cd_budget = (TIME_LIMIT - 30) / n_cd_restarts
    
    for restart in range(n_cd_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        if restart == 0:
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            f0 = np.exp(-8 * x**2)
        elif restart == 1:
            # Flat with slight modulation
            f0 = 1.0 + 0.3 * np.cos(2 * np.pi * 3 * np.arange(N) / N)
            f0 = np.maximum(f0, 0.01)
        elif restart == 2:
            # Two bumps
            x = np.linspace(-0.25, 0.25, N, endpoint=False)
            f0 = np.exp(-30 * (x - 0.1)**2) + np.exp(-30 * (x + 0.1)**2)
        else:
            # Random smooth
            f0 = np.abs(rng.normal(0, 1, N)) + 0.1
            f0 = gaussian_filter1d(f0, sigma=8)
        
        f0 = f0 / np.sum(f0)
        
        t_end = min(start_time + cd_budget, TIME_LIMIT - 30)
        f_res, score_res = spectral_coordinate_descent(f0, start_time, t_end, n_iters=3000)
        
        if score_res < best_score:
            best_score = score_res
            best_f = f_res
    
    # Phase 2: Spectral annealing to explore different phase configurations
    if time.time() - start_time < TIME_LIMIT - 20:
        t_end = TIME_LIMIT - 15
        f_res, score_res = spectral_annealing(best_f, start_time, t_end)
        
        if score_res < best_score:
            best_score = score_res
            best_f = f_res
    
    # Phase 3: Local refinement in time domain
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 2
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
