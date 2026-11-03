"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a parametric family based on |sum of Gaussian modes|^2 for global
exploration via Nelder-Mead, followed by aggressive FFT-based local refinement
with multi-peak sensitivity and adaptive step sizes.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill

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

def fft_autoconv(f):
    """Compute autoconvolution via FFT."""
    fft_size = 2 * N
    f_fft = np.fft.rfft(f, n=fft_size)
    conv_fft = f_fft * f_fft
    conv = np.fft.irfft(conv_fft, n=fft_size)
    return conv[:2*N-1]

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def parametric_to_step(params, n_modes):
    """Convert parameters to a step function of length N.
    
    params layout: [centers(n_modes), sigmas(n_modes), re_coeffs(n_modes), im_coeffs(n_modes)]
    Total: 4 * n_modes parameters
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    centers = params[:n_modes]
    sigmas = np.abs(params[n_modes:2*n_modes]) + 0.005
    sigmas = np.minimum(sigmas, 0.25)
    re_coeffs = params[2*n_modes:3*n_modes]
    im_coeffs = params[3*n_modes:4*n_modes]
    
    signal = np.zeros(N, dtype=np.complex128)
    for k in range(n_modes):
        g = np.exp(-0.5 * ((x - centers[k]) / sigmas[k]) ** 2)
        signal += (re_coeffs[k] + 1j * im_coeffs[k]) * g
    
    f = np.abs(signal) ** 2
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_modes, beta=10.0):
    f = parametric_to_step(params, n_modes)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return smooth_max(conv, beta) / (s * s)

def fast_peak_shaving(h, start_time, max_time):
    """Aggressive local refinement using FFT-based convolution and sensitivity-guided steps.
    
    Targets the maximum of the autoconvolution by reducing bins that contribute
    most to the peak, while maintaining non-negativity and constant sum.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    patience = 0
    max_patience = 1500
    n_peaks = 8  # consider top n_peaks lags
    
    fft_size = 2 * N
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Compute autoconvolution using FFT
        conv = fft_autoconv(current_h)
        
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find indices of top peaks
        peak_indices = np.argsort(conv)[-n_peaks:]
        
        # Compute combined gradient from all peak lags
        g = np.zeros(N)
        for k_star in peak_indices:
            j_min = max(0, k_star - (N - 1))
            j_max = min(N - 1, k_star)
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                f_indices = k_star - indices
                g[indices] += current_h[f_indices]
        
        # Center g to preserve sum
        g_mean = np.mean(g)
        g = g - g_mean
        
        # Adaptive step size
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        h_scale = np.mean(current_h)
        alpha_effective = alpha * h_scale / g_norm
        
        # Update
        h_new = current_h - alpha_effective * g
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Check improvement
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val - 1e-15:
            current_h = h_new
            current_val = val_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.92
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """High-precision polish: very small steps with coordinate-wise perturbations."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.005
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val - 1e-15:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Find peak
        k_star = np.argmax(conv)
        
        # Compute sensitivity vector
        g = np.zeros(N)
        j_min = max(0, k_star - (N - 1))
        j_max = min(N - 1, k_star)
        if j_min <= j_max:
            indices = np.arange(j_min, j_max + 1)
            f_indices = k_star - indices
            g[indices] = current_h[f_indices]
        
        # Try perturbations on high-sensitivity coordinates
        sorted_idx = np.argsort(g)[::-1]
        
        improved = False
        n_trials = min(30, N)
        for trial in range(n_trials):
            if time.time() - start_time > max_time:
                break
            
            i_high = sorted_idx[trial % N]
            # Choose a low-sensitivity target
            i_low = (i_high + N // 2 + trial) % N
            
            if current_h[i_high] < 1e-10:
                continue
            
            delta = alpha * current_h[i_high]
            if delta < 1e-12:
                continue
            
            h_test = current_h.copy()
            h_test[i_high] -= delta
            h_test[i_low] += delta
            h_test = np.maximum(h_test, 0.0)
            
            s_test = np.sum(h_test)
            if s_test < 1e-12:
                continue
            h_test = h_test * (N / s_test)
            
            conv_test = fft_autoconv(h_test)
            val_test = np.max(conv_test) / (N * N)
            
            if val_test < current_val - 1e-15:
                current_h = h_test
                current_val = val_test
                improved = True
        
        if not improved:
            alpha *= 0.85
            if alpha < 1e-15:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try Gaussian-mode parametric optimization with multiple mode counts
    for n_modes in [2, 3, 4, 5, 6]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(100 + n_modes)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
            
            for restart in range(10):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_modes)
                sigmas = rng.uniform(0.05, 0.15, size=n_modes)
                re_coeffs = rng.normal(0, 1.0, size=n_modes)
                im_coeffs = rng.normal(0, 1.0, size=n_modes)
                
                x0 = np.concatenate([centers, sigmas, re_coeffs, im_coeffs])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_modes, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x.copy()
                except Exception:
                    pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_modes)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Also try cosine bump as baseline
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cosine = np.cos(np.pi * x) ** 2
    f_cosine = np.maximum(f_cosine, 0)
    cos_score = compute_score(f_cosine)
    
    if cos_score < best_score:
        best_f = f_cosine
        best_score = cos_score
    
    # Local refinement: Fast peak shaving
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = fast_peak_shaving(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # High-precision polish pass
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 1
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
