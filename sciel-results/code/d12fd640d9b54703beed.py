"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction.
4. Add a final polish pass with tighter tolerances.

Key improvements:
- Better initial parameter sampling with structured diversity
- More aggressive local refinement with adaptive step sizes
- Final polish pass to squeeze out remaining improvements
- FFT-based autoconvolution for speed, enabling more iterations
- Improved sensitivity computation using FFT cross-correlation
- Constant function as guaranteed lower bound
- Additional local refinement iterations with adaptive alpha
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

def fft_autocorr(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    # Use next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:size]
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
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_fft(current_h, peak_indices):
    """Compute sensitivity using FFT cross-correlation.
    
    For each peak lag m, sensitivity[j] += current_h[m - j] for valid j.
    This is a cross-correlation of current_h with an indicator of peak lags.
    """
    N = len(current_h)
    # Create peak indicator
    peak_ind = np.zeros(2 * N - 1)
    peak_ind[peak_indices] = 1.0
    
    # sensitivity[j] = sum_{m in peaks} current_h[m-j] for valid j
    # This is: sensitivity = cross-correlation of current_h with peak_ind
    # Using FFT: sensitivity[j] = sum_m peak_ind[m] * current_h[m-j]
    # = (current_h * peak_ind_flipped)[j] where * is convolution
    # Let's use FFT for efficiency
    
    # We want: sensitivity[j] = sum_{m=0}^{2N-2} peak_ind[m] * current_h[m-j]
    # for j where 0 <= m-j < N, i.e., j <= m < j+N
    # This is equivalent to: sensitivity = np.correlate(current_h, peak_ind, mode='valid')
    # but we need to be careful about the indexing
    
    # Actually, let's use the fact that:
    # sensitivity[j] = sum_{k=0}^{N-1} current_h[k] * peak_ind[j+k]
    # This is a cross-correlation: sensitivity = np.correlate(peak_ind, current_h, mode='valid')
    # but we need to check the indexing
    
    # Let's use FFT approach:
    # sensitivity[j] = sum_m peak_ind[m] * current_h[m-j]
    # Let's define: for each j, sum over m where 0<=m<2N-1 and 0<=m-j<N
    # m-j >= 0 => m >= j
    # m-j < N => m < j+N
    # So m ranges from j to min(j+N-1, 2N-2)
    
    # Using FFT: 
    # Let's compute the full cross-correlation using FFT
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    H = np.fft.rfft(current_h, n=fft_size)
    P = np.fft.rfft(peak_ind, n=fft_size)
    
    # Cross-correlation: corr[j] = sum_k current_h[k] * peak_ind[j+k]
    # In frequency domain: corr = ifft(conj(H) * P)
    # But we need to be careful about the indexing
    
    # Actually, let's use a simpler approach:
    # sensitivity[j] = sum_{m in peaks} current_h[m-j] for valid m-j
    # This is the same as: for each peak m, add current_h[m-j] to sensitivity[j]
    # where j ranges from max(0, m-(N-1)) to min(N-1, m)
    
    # Let's vectorize this using a different approach:
    # Create a matrix where each row corresponds to a peak index
    # But that might be memory intensive
    
    # Better: use the fact that sensitivity is a convolution-like operation
    # Let's flip current_h and convolve with peak_ind
    # sensitivity[j] = sum_m peak_ind[m] * current_h[m-j]
    # = sum_m peak_ind[m] * current_h_flipped[N-1-(m-j)]
    # where current_h_flipped[k] = current_h[N-1-k]
    
    # Let's just do it with a vectorized loop over peaks but using numpy slicing
    sensitivity = np.zeros(N)
    for m in peak_indices:
        j_min = max(0, m - (N - 1))
        j_max = min(N - 1, m)
        if j_max >= j_min:
            idx = m - np.arange(j_min, j_max + 1)
            sensitivity[j_min:j_max+1] += current_h[idx]
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    # Adaptive threshold: start wider, tighten over time
    iter_count = 0
    total_iters_estimate = 200  # rough estimate
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Adaptive threshold: start at 0.90, tighten to 0.97
        progress = min(iter_count / total_iters_estimate, 1.0)
        threshold_frac_adaptive = 0.90 + 0.07 * progress
        threshold = max_c * threshold_frac_adaptive
        peak_indices = np.where(conv >= threshold)[0]
        
        # If no peaks found (should not happen), fallback to max index
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        # Compute sensitivity
        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
        # Normalize sensitivity
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        # Use a slightly damped update to prevent oscillation
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
            alpha = min(alpha * 1.05, 0.5) # Slightly more aggressive growth
        else:
            alpha *= 0.85 # Slightly faster decay
            if alpha < 1e-8:
                break
        
        iter_count += 1
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    # Use a more aggressive initial alpha for polish
    alpha = 0.03
    iter_count = 0
    total_iters_estimate = 100
    
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
        
        # Very tight threshold for polish, tighten over time
        progress = min(iter_count / total_iters_estimate, 1.0)
        threshold_frac = 0.99 + 0.009 * progress
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        if len(peak_indices) == 0:
            peak_indices = np.array([np.argmax(conv)])

        sensitivity = compute_sensitivity_fft(current_h, peak_indices)
        
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
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.90
            if alpha < 1e-10:
                break
        
        iter_count += 1
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with better initialization
    # Reduced time budget for parametric optimization to allow more local refinement
    for n_c in [3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 70:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [10.0, 20.0, 30.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 50:
                break
            
            for restart in range(5):
                if time.time() - start_time > TIME_LIMIT - 45:
                    break
                
                # Better structured initialization
                if restart == 0:
                    # Symmetric initialization
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    # Asymmetric with varying widths
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                elif restart == 2:
                    # Concentrated bumps
                    centers = rng.uniform(-0.1, 0.1, size=n_c)
                    widths = rng.uniform(0.03, 0.15, size=n_c)
                    heights = rng.uniform(1.0, 3.0, size=n_c)
                else:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.03, 0.3, size=n_c)
                    heights = rng.uniform(0.3, 2.5, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
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
    
    # Local refinement: sensitivity-based peak reduction
    # Increased time budget
    if time.time() - start_time < TIME_LIMIT - 25:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final polish pass
    if time.time() - start_time < TIME_LIMIT - 8:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 3
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
