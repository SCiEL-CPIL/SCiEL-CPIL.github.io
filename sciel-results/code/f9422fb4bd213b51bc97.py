"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use a smooth parametric family based on cosine bumps, then refine
with a sensitivity-based local optimization that targets peak reduction.

The key insight is to use a family of functions whose autoconvolution has a
flatter peak. We use a smooth max (log-sum-exp) surrogate for the parametric
optimization, then apply a gradient-like local refinement that reduces the
sensitivity of peak lags to reduce the maximum autoconvolution value.
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

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    log_sum = x_max + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum

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

def spectral_coordinate_descent(h, start_time, max_time):
    """Spectral (FFT-domain) coordinate descent for peak autoconvolution reduction.
    
    Works in the frequency domain: modify individual frequency bins to reduce
    the peak of the autoconvolution. The autoconvolution is |F|^2 in frequency,
    so we can adjust phases/magnitudes of frequency components.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    alpha = 0.01
    no_improve = 0
    max_no_improve = 20
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Find peak lags
        threshold = max_c * 0.97
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # For each frequency bin, compute how much reducing its magnitude
        # would reduce the peak autoconvolution.
        # The contribution of bin k to conv is |f_hat[k]|^2 * exp(2j*pi*k*n/n_fft)
        # We want to reduce bins that align with peak lags.
        
        # Compute the "peak-aligned" component of each frequency bin
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        
        # Sensitivity: how much each frequency bin contributes to peak lags
        # = Re(f_hat[k] * conj(pm_hat[k])) / |f_hat[k]|  (phase alignment)
        magnitudes = np.abs(f_hat)
        phase_align = np.real(f_hat * np.conj(pm_hat)) / (magnitudes + 1e-15)
        
        # Reduce magnitude of bins with high positive phase alignment
        # and increase bins with negative alignment (to maintain sum)
        sens = phase_align - np.mean(phase_align)
        sens = np.clip(sens, -1.0, 1.0)
        
        # Adjust magnitudes
        new_mags = magnitudes * (1.0 - alpha * sens)
        new_mags = np.maximum(new_mags, 0.0)
        
        # Preserve phases
        phases = np.angle(f_hat)
        new_f_hat = new_mags * np.exp(1j * phases)
        
        # Ensure DC component maintains normalization
        new_h = np.fft.irfft(new_f_hat, n=n_fft)[:N]
        new_h = np.maximum(new_h, 0.0)
        
        s_new = np.sum(new_h)
        if s_new < 1e-12:
            break
        new_h = new_h * (N / s_new)
        
        # Evaluate
        new_f_hat2 = np.fft.rfft(new_h, n=n_fft)
        conv_new = np.fft.irfft(new_f_hat2 * new_f_hat2, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = new_h
            alpha = min(alpha * 1.02, 0.1)
            no_improve = 0
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction.
    
    Uses FFT to efficiently compute sensitivity of each bin to peak lags,
    then reduces bins that disproportionately contribute to the maximum
    autoconvolution values. Uses log-sum-exp smoothing for more robust
    gradient signal.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    n_fft = 2 * N  # size for full convolution via FFT
    no_improve = 0
    max_no_improve = 50
    beta_lse = 20.0  # log-sum-exp sharpness
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Use FFT for fast convolution
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Find lags where conv is close to max
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        # Also compute a soft-weighted peak mask using log-sum-exp inspired weights
        # This gives smoother gradients than a hard mask
        conv_shifted = conv - max_c
        soft_weights = np.exp(beta_lse * conv_shifted / max_c)
        soft_weights = soft_weights / (np.sum(soft_weights) + 1e-15)
        # Combine hard and soft masks
        combined_mask = 0.5 * peak_mask + 0.5 * soft_weights * (2*N-1)
        combined_mask = combined_mask / (np.sum(combined_mask) + 1e-15)
        
        # Compute sensitivity via FFT:
        # sensitivity[j] = sum_m combined_mask[m] * current_h[m-j] (where valid)
        # This is a cross-correlation of combined_mask with current_h
        pm_hat = np.fft.rfft(combined_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        # Cross-correlation: ifft(fft(pm) * conj(fft(ch)))
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        # Normalize sensitivity
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        # Reduce bins with high sensitivity (they contribute to peaks)
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Evaluate new candidate
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.3)
            threshold_frac = max(threshold_frac - 0.005, 0.85)
            beta_lse = min(beta_lse * 1.02, 50.0)
        else:
            alpha *= 0.9
            threshold_frac = min(threshold_frac + 0.002, 0.98)
            beta_lse = max(beta_lse * 0.98, 10.0)
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: use a raised cosine (better than constant for this problem)
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    best_f = 0.5 * (1.0 + np.cos(2.0 * np.pi * x))
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    best_f = best_f * (N / s)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components with tighter time budget
    # Reserve most time for local refinement
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.20
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Fewer restarts, tighter tolerances to save time
        betas = [10.0, 20.0]
        
        for beta in betas:
            if time.time() > param_budget_end:
                break
            
            for restart in range(3):
                if time.time() > param_budget_end:
                    break
                
                centers = rng.uniform(-0.15, 0.15, size=n_c)
                widths = rng.uniform(0.08, 0.25, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-7}
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
    
    # Local refinement: FFT-based sensitivity peak reduction
    if time.time() < TIME_LIMIT - 40:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 40
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Spectral coordinate descent: different optimization path in frequency domain
    if time.time() < TIME_LIMIT - 20:
        spec_f, spec_score = spectral_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 20
        )
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Final high-precision polish: many small steps with very tight threshold
    if time.time() < TIME_LIMIT - 10:
        best_f = best_f.copy()
        alpha_small = 0.01
        n_fft = 2 * N
        for _ in range(200):
            if time.time() > TIME_LIMIT - 10:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Very tight threshold: only the true peak
            threshold = max_c * 0.995
            peak_mask = (conv >= threshold).astype(np.float64)
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            h_new = best_f * (1.0 - alpha_small * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                best_f = h_new
                alpha_small = min(alpha_small * 1.01, 0.05)
            else:
                alpha_small *= 0.95
                if alpha_small < 1e-5:
                    break
    
    # Ultimate polish: pairwise bin adjustments targeting the exact peak
    if time.time() < TIME_LIMIT - 2:
        polished_f, polished_score = _high_precision_polish(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]


def _high_precision_polish(f, start_time, max_time):
    """Very fine-grained local search using proper gradient at peak lag."""
    best_f = f.copy()
    best_val = compute_score(f)
    n_fft = 2 * N
    delta = 0.005
    
    while time.time() - start_time < max_time:
        s = np.sum(best_f)
        if s < 1e-12:
            break
        best_f = best_f * (N / s)
        
        f_hat = np.fft.rfft(best_f, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
        
        # Find the argmax lag
        peak_lag = np.argmax(conv)
        
        # Proper gradient: d(conv[k])/d(f[i]) = f[k-i] for valid indices
        # conv[k] = sum_i f[i]*f[k-i], so d(conv[k])/d(f[i]) = f[k-i]
        sens = np.zeros(N)
        for i in range(N):
            j = peak_lag - i
            if 0 <= j < N:
                sens[i] = best_f[j]
        
        # Also consider second-order: reduce bins where both f[i] and f[peak-i] are large
        # This targets the product terms that create the peak
        sens2 = np.zeros(N)
        for i in range(N):
            j = peak_lag - i
            if 0 <= j < N:
                sens2[i] = best_f[i] * best_f[j]
        
        # Combine: primary gradient + secondary product term
        sens_combined = sens + 0.5 * sens2
        sens_max = np.max(sens_combined) + 1e-15
        sens_norm = sens_combined / sens_max
        
        # Reduce bins that contribute most to the peak
        h_new = best_f * (1.0 - delta * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        f_new_hat = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            best_f = h_new
            delta = min(delta * 1.02, 0.02)
        else:
            delta *= 0.9
            if delta < 1e-6:
                break
    
    return best_f, best_val

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
