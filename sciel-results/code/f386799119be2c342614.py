"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on cosine bumps with multiple components.
2. Optimize using Nelder-Mead with a smooth max surrogate.
3. Apply local refinement using FFT-accelerated sensitivity-based peak reduction.
4. Add discrete annealing pass for fine-tuning.
5. Final polish pass with tighter tolerances.
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
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for autoconvolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
    return np.max(conv) / (s * s)

def fft_autoconv(f):
    """Compute autoconvolution using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2*N-1]
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
    conv = fft_autoconv(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Epigraph formulation: minimize t such that conv[i] <= t * s^2 for all i
    # Smooth surrogate: t = max(conv) / s^2 approximated via log-sum-exp
    return smooth_max(conv, beta) / (s * s)

def objective_epigraph(params, n_components, beta=20.0, t_init=0.1):
    """Epigraph-style objective with auxiliary variable for smoother landscape."""
    f = parametric_to_step(params, n_components)
    conv = fft_autoconv(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use a tighter smooth max to better approximate the true max
    return smooth_max(conv, beta) / (s * s)

def compute_sensitivity_fft(h, peak_indices):
    """Compute sensitivity using FFT correlation."""
    sensitivity = np.zeros(N)
    for m in peak_indices:
        j_min = max(0, m - (N - 1))
        j_max = min(N - 1, m)
        if j_min <= j_max:
            sensitivity[j_min:j_max+1] += h[m - np.arange(j_min, j_max+1)]
    return sensitivity

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.98):
    """Local refinement using FFT-accelerated sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
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
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
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
        
        conv_new = fft_autoconv(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break
    
    # Spectral refinement: modify in frequency domain to reduce peak
    if time.time() - start_time < max_time - 5:
        best_h = best_h * (N / np.sum(best_h))
        conv = fft_autoconv(best_h)
        max_c = np.max(conv)
        best_val = max_c / (N * N)
        
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        F = np.fft.rfft(best_h, n=n_fft)
        
        alpha_spec = 0.05
        no_improve_spec = 0
        for _ in range(80):
            if time.time() - start_time > max_time - 3:
                break
            if no_improve_spec > 30:
                break
            
            conv = fft_autoconv(best_h)
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            if current_val < best_val:
                best_val = current_val
                no_improve_spec = 0
            else:
                no_improve_spec += 1
            
            # Find all significant peak lags
            peak_threshold = max_c * 0.95
            peak_lags = np.where(conv >= peak_threshold)[0]
            
            if len(peak_lags) == 0:
                peak_lags = np.array([np.argmax(conv)])
            
            # Compute average phase alignment across all peak lags
            k_max = n_fft // 2
            freqs = np.arange(k_max)
            
            # For each peak lag, compute which frequencies constructively contribute
            # and average the sensitivity across peaks
            avg_phase_align = np.zeros(k_max)
            for lag in peak_lags[:10]:  # Limit to top 10 peaks for efficiency
                avg_phase_align += np.cos(2 * np.pi * freqs * lag / n_fft)
            avg_phase_align /= min(len(peak_lags), 10)
            
            # Reduce magnitudes where phase aligns positively (constructive interference)
            F_modified = F.copy()
            reduction = 1.0 - alpha_spec * np.abs(avg_phase_align[1:k_max])
            reduction = np.clip(reduction, 0.5, 1.0)
            F_modified[1:k_max] *= reduction
            
            # Ensure real-valued, non-negative result
            h_candidate = np.fft.irfft(F_modified, n=n_fft)[:N]
            h_candidate = np.maximum(h_candidate, 0.0)
            s_cand = np.sum(h_candidate)
            if s_cand < 1e-12:
                alpha_spec *= 0.5
                continue
            h_candidate = h_candidate * (N / s_cand)
            
            conv_cand = fft_autoconv(h_candidate)
            val_cand = np.max(conv_cand) / (N * N)
            
            if val_cand < current_val:
                best_h = h_candidate
                F = F_modified
                alpha_spec = min(alpha_spec * 1.05, 0.25)
                no_improve_spec = 0
            else:
                alpha_spec *= 0.85
                if alpha_spec < 1e-6:
                    break
    
    return best_h, best_val

def discrete_annealing(h, start_time, max_time):
    """Discrete annealing: flip individual bins to minimize peak autoconvolution."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    current_val = best_val
    
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h * (N / s)
    
    T = 0.3
    cooling = 0.998
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < max_time and T > 0.001:
        j = rng.randint(N)
        old_val = current_h[j]
        
        # Smaller perturbations for finer tuning
        delta = rng.normal(0, 0.05 * max(old_val, 0.05))
        new_val = max(0.0, old_val + delta)
        
        if new_val == old_val:
            continue
        
        current_h[j] = new_val
        s_new = np.sum(current_h)
        if s_new < 1e-12:
            current_h[j] = old_val
            continue
        current_h = current_h * (N / s_new)
        
        conv_new = fft_autoconv(current_h)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val or rng.random() < np.exp(-(val_new - current_val) / T):
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_h = current_h.copy()
        else:
            current_h[j] = old_val
        
        T *= cooling
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass with very small step sizes and multi-scale sensitivity."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.012
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() - start_time < max_time and no_improve_count < max_no_improve:
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
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Multi-scale peak detection: use multiple thresholds
        # Tight threshold for precise peak location
        threshold_tight = max_c * 0.999
        # Medium threshold for broader context
        threshold_med = max_c * 0.99
        # Loose threshold for global structure
        threshold_loose = max_c * 0.95
        
        peak_tight = np.where(conv >= threshold_tight)[0]
        peak_med = np.where(conv >= threshold_med)[0]
        peak_loose = np.where(conv >= threshold_loose)[0]
        
        if len(peak_tight) == 0:
            peak_tight = np.array([np.argmax(conv)])
        if len(peak_med) == 0:
            peak_med = peak_tight
        if len(peak_loose) == 0:
            peak_loose = peak_med
        
        # Compute multi-scale sensitivity
        sens_tight = compute_sensitivity_fft(current_h, peak_tight)
        sens_med = compute_sensitivity_fft(current_h, peak_med)
        sens_loose = compute_sensitivity_fft(current_h, peak_loose)
        
        # Weighted combination: emphasize tight peaks more
        sensitivity = 0.5 * sens_tight + 0.3 * sens_med + 0.2 * sens_loose
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Multi-step trial: try several alpha values with finer granularity
        best_trial_h = current_h
        best_trial_val = current_val
        for trial_alpha in [alpha * 0.3, alpha * 0.5, alpha * 0.75, alpha, alpha * 1.5, alpha * 2.0]:
            h_trial = current_h * (1.0 - trial_alpha * sens_norm)
            h_trial = np.maximum(h_trial, 0.0)
            s_trial = np.sum(h_trial)
            if s_trial < 1e-12:
                continue
            h_trial = h_trial * (N / s_trial)
            conv_trial = fft_autoconv(h_trial)
            val_trial = np.max(conv_trial) / (N * N)
            if val_trial < best_trial_val:
                best_trial_val = val_trial
                best_trial_h = h_trial
        
        if best_trial_val < current_val:
            current_h = best_trial_h
            alpha = min(alpha * 1.03, 0.08)
        else:
            alpha *= 0.93
            if alpha < 1e-13:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    for n_c in [3, 4, 5, 6, 7, 8]:
        if time.time() - start_time > TIME_LIMIT - 55:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        betas = [5.0, 10.0, 20.0, 30.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 35:
                break
            
            for restart in range(8):
                if time.time() - start_time > TIME_LIMIT - 28:
                    break
                
                if restart == 0:
                    centers = np.linspace(-0.15, 0.15, n_c)
                    widths = np.full(n_c, 0.15)
                    heights = np.full(n_c, 1.0)
                elif restart == 1:
                    centers = rng.uniform(-0.2, 0.2, size=n_c)
                    widths = rng.uniform(0.05, 0.25, size=n_c)
                    heights = rng.uniform(0.5, 2.0, size=n_c)
                elif restart == 2:
                    # Symmetric pattern
                    half_n = n_c // 2
                    centers = np.concatenate([
                        np.linspace(-0.18, -0.02, half_n),
                        np.linspace(0.02, 0.18, n_c - half_n)
                    ])
                    widths = rng.uniform(0.04, 0.2, size=n_c)
                    heights = rng.uniform(0.5, 1.5, size=n_c)
                else:
                    centers = rng.uniform(-0.22, 0.22, size=n_c)
                    widths = rng.uniform(0.02, 0.35, size=n_c)
                    heights = rng.uniform(0.2, 3.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
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
    
    if time.time() - start_time < TIME_LIMIT - 30:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 15
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    if time.time() - start_time < TIME_LIMIT - 12:
        annealed_f, annealed_score = discrete_annealing(
            best_f, start_time, TIME_LIMIT - 6
        )
        if annealed_score < best_score:
            best_f = annealed_f
            best_score = annealed_score
    
    # Run polish twice for better convergence
    if time.time() - start_time < TIME_LIMIT - 4:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Second polish pass with remaining time
    if time.time() - start_time < TIME_LIMIT - 1:
        polished_f2, polished_score2 = polish_pass(
            best_f, start_time, TIME_LIMIT - 0.3
        )
        if polished_score2 < best_score:
            best_f = polished_f2
            best_score = polished_score2
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
