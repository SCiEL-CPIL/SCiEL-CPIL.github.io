"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction.
4. Add a final polish pass with tighter tolerances.

Key improvements over the seed:
- Better initial parameter sampling with structured diversity
- More aggressive local refinement with adaptive step sizes
- Final polish pass to squeeze out remaining improvements
"""

import numpy as np
from scipy.optimize import minimize
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

def fft_autoconv(f):
    """Compute autoconvolution via FFT for speed."""
    n_conv = 2 * N - 1
    n_fft = 1 << (n_conv - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:n_conv]
    return conv

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
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_vectorized(f, peak_indices):
    """Vectorized computation of sensitivity: for each bin j, sum of f[m-j] over peak lags m."""
    # conv[m] = sum_j f[j] * f[m-j]
    # sensitivity[j] = sum_{m in peaks} f[m-j]
    # This is equivalent to cross-correlation of f with the peak indicator
    peak_indicator = np.zeros(2 * N - 1)
    peak_indicator[peak_indices] = 1.0
    # sensitivity[j] = sum_m peak_indicator[m] * f[m-j]
    # = (peak_indicator * f)[j] in the appropriate range
    # Actually: sensitivity[j] = sum_m PI[m] * f[m-j] for valid m,j
    # This is the cross-correlation of PI with f, evaluated at lags 0..N-1
    # Use FFT: sensitivity = ifft(FFT(PI) * conj(FFT(f)))
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=n_fft)
    PI = np.fft.rfft(peak_indicator, n=n_fft)
    # Cross-correlation: ifft(conj(F) * PI) gives sum_j f[j] * PI[j+k]
    # We want sum_m PI[m] * f[m-j] = sum_k PI[k+j] * f[k] = cross_corr at lag j
    # cross_corr[j] = sum_k f[k] * PI[k+j]
    cc = np.fft.irfft(np.conj(F) * PI, n=n_fft)
    # cc[j] for j=0..N-1 is what we want (with proper indexing)
    sensitivity = cc[:N].copy()
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.90):
    """Local refinement using sensitivity-based peak reduction (vectorized) with adaptive threshold."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    n_iterations = 0
    stall_count = 0
    
    while time.time() - start_time < max_time and stall_count < 8:
        n_iterations += 1
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Adaptive threshold: tighten as we progress
        threshold = max_c * (threshold_frac + 0.05 * min(n_iterations / 50.0, 1.0))
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) < 3:
            peak_indices = np.where(conv >= max_c * 0.95)[0]
        
        # Vectorized sensitivity computation
        sensitivity = compute_sensitivity_vectorized(current_h, peak_indices)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.45)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes (vectorized)."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.015
    stall_count = 0
    
    while time.time() - start_time < max_time and stall_count < 10:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val - 1e-12:
            best_val = current_val
            best_h = current_h.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Very tight threshold for polish
        threshold = max_c * 0.995
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) < 2:
            peak_indices = np.where(conv >= max_c * 0.99)[0]
        
        # Vectorized sensitivity
        sensitivity = compute_sensitivity_vectorized(current_h, peak_indices)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.12)
        else:
            alpha *= 0.88
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def coordinate_descent(h, start_time, max_time):
    """Exact 1D coordinate descent using scipy.optimize.minimize_scalar."""
    from scipy.optimize import minimize_scalar
    
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    s = np.sum(current_h)
    current_h = current_h * (N / s)  # normalize
    
    n_rounds = 0
    no_improve_count = 0
    
    while time.time() - start_time < max_time and no_improve_count < 2:
        n_rounds += 1
        round_improved = False
        
        for j in range(N):
            if time.time() - start_time > max_time - 2:
                break
            
            old_val = current_h[j]
            
            # Define objective for 1D minimization: change f[j], renormalize, compute score
            def obj_1d(x_new):
                if x_new < 0:
                    x_new = 0.0
                current_h[j] = x_new
                s_new = np.sum(current_h)
                if s_new < 1e-12:
                    return np.inf
                h_temp = current_h * (N / s_new)
                conv_temp = fft_autoconv(h_temp)
                return np.max(conv_temp) / (N * N)
            
            # Bound the search: try [0, 3*old_val]
            upper = max(3.0 * old_val, 0.1)
            
            try:
                result = minimize_scalar(
                    obj_1d,
                    bounds=(0.0, upper),
                    method='bounded',
                    options={'maxiter': 30, 'xatol': 1e-8}
                )
                if result.fun < best_val - 1e-10:
                    current_h[j] = result.x
                    s_new = np.sum(current_h)
                    current_h = current_h * (N / s_new)
                    best_val = result.fun
                    best_h = current_h.copy()
                    round_improved = True
                else:
                    current_h[j] = old_val
            except Exception:
                current_h[j] = old_val
        
        if round_improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric cosine bump optimization (reduced budget, fewer restarts)
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 55:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [10.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 40:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 35:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
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
    
    # Phase 2: Local refinement (vectorized, fast) - give it more time
    if time.time() - start_time < TIME_LIMIT - 25:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 18
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 3: Coordinate descent on full space (exact 1D)
    if time.time() - start_time < TIME_LIMIT - 12:
        cd_f, cd_score = coordinate_descent(
            best_f, start_time, TIME_LIMIT - 6
        )
        if cd_score < best_score:
            best_f = cd_f
            best_score = cd_score
    
    # Phase 4: Final polish pass (vectorized)
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
