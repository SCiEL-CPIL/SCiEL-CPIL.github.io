"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on Cosine bumps (compact support, C1 smooth).
2. Optimize using Nelder-Mead (derivative-free) to avoid gradient issues.
3. Apply local refinement using sensitivity-based peak reduction with FFT.
4. Add a final polish pass with adaptive step sizes.

Key improvements:
- Cosine bumps instead of Gaussians: Compact support reduces tail interference in autoconvolution.
- Nelder-Mead optimization: Robust against non-smoothness in parametric mapping.
- FFT-based convolution for speed.
- Simulated annealing noise in local refinement to escape local minima.
- Spectral refinement to flatten power spectrum and reduce peak autoconvolution.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

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

def cosine_basis(x, centers, widths, heights):
    """
    Generate a smooth nonnegative function from Cosine bump components.
    Each bump is h * cos^2(pi * (x - c) / w) for |x - c| <= w/2, else 0.
    """
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        # Ensure width is positive
        w_eff = np.abs(w) + 1e-6
        mask = dist <= w_eff / 2.0
        if np.any(mask):
            # cos^2 is non-negative and smooth (C1) at the boundaries
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w_eff) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    # Enforce constraints via transformation
    # Widths must be > 0. We use abs + small epsilon.
    # Heights must be >= 0. We use abs.
    # Centers are free in [-0.25, 0.25] roughly, but we let optimizer explore.
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=20.0):
    """Objective function for Nelder-Mead. Uses smooth max surrogate."""
    f = parametric_to_step(params, n_components)
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    return smooth_max(conv, beta) / (s * s)

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT and simulated annealing noise."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity using FFT-based approach
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Add noise for simulated annealing effect
        noise = rng.normal(0, 0.1, size=N)
        
        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm + alpha * 0.5 * noise)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.04, 0.5)
        else:
            delta = val_new - current_val
            if delta > 0:
                # Accept with probability exp(-delta/alpha)
                accept_prob = np.exp(-delta / (alpha + 1e-12))
                if rng.random() < accept_prob:
                    current_h = h_new
            alpha *= 0.95
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def spectral_refinement(h, start_time, max_time):
    """Refine using frequency domain insights to flatten power spectrum."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    for iteration in range(500):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        current_val = np.max(conv) / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Get FFT
        F = np.fft.rfft(current_h, n=2*N)
        power = np.abs(F) ** 2
        
        # Identify frequency bins with high power that contribute to peak
        # The DC component is always present; focus on others
        if len(power) > 1:
            power_no_dc = power[1:]
            max_power = np.max(power_no_dc)
            if max_power > 1e-12:
                # Reduce high-power frequency components
                threshold = max_power * 0.9
                high_power_mask = power_no_dc >= threshold
                # Dampen these components in frequency domain
                F_damped = F.copy()
                F_damped[1:][high_power_mask] *= 0.98
                # Inverse FFT
                h_new = np.fft.irfft(F_damped, n=2*N)[:N]
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new > 1e-12:
                    h_new = h_new * (N / s_new)
                    conv_new = fftconvolve(h_new, h_new, mode='full')
                    val_new = np.max(conv_new) / (N * N)
                    if val_new < current_val:
                        current_h = h_new
                    else:
                        break
                else:
                    break
        else:
            break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fftconvolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                sensitivity[j_min:j_max+1] += current_h[m - j_max:m - j_min + 1]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fftconvolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.2)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of Cosine bump components
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Nelder-Mead settings
        # We use multiple restarts with different initializations
        n_restarts = 12
        
        for restart in range(n_restarts):
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            # Structured initialization
            if restart == 0:
                # Symmetric initialization
                centers = np.linspace(-0.15, 0.15, n_c)
                widths = np.full(n_c, 0.12)
                heights = np.full(n_c, 1.0)
            elif restart == 1:
                # Asymmetric with varying widths
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.04, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
            elif restart == 2:
                # Concentrated at center
                centers = np.zeros(n_c) + rng.uniform(-0.05, 0.05, size=n_c)
                widths = rng.uniform(0.03, 0.1, size=n_c)
                heights = rng.uniform(0.8, 2.5, size=n_c)
            else:
                # Random
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.02, 0.25, size=n_c)
                heights = rng.uniform(0.3, 3.0, size=n_c)
            
            x0 = np.concatenate([centers, widths, heights])
            
            try:
                result = minimize(
                    objective, x0, args=(n_c, 20.0),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is not None:
            f = parametric_to_step(best_params, n_c)
            actual_score = compute_score(f)
            if actual_score < best_score:
                best_f = f
                best_score = actual_score
    
    # Local refinement: sensitivity-based peak reduction with simulated annealing
    if time.time() - start_time < TIME_LIMIT - 30:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral refinement to flatten power spectrum
    if time.time() - start_time < TIME_LIMIT - 10:
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 2:
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
