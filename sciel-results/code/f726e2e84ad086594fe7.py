"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Parametric initialization with cosine bumps (smooth family)
2. FFT-based local refinement using smooth peak masks and backtracking line search
3. Haar wavelet coordinate descent to exploit piecewise-constant structure
   and escape local minima of time-domain methods
4. Final high-precision polish

The key insight is to combine the smooth sensitivity-based gradient approach
(FFT-accelerated) with a greedy coordinate descent in the Haar wavelet basis.
The Haar basis is natural for piecewise-constant functions and the orthogonality
allows us to make independent adjustments that reduce peak autoconvolution
without immediately undoing previous improvements.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 148  # safety margin before 150s kill

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

def fft_convolve(f):
    """Fast autoconvolution via FFT."""
    n_fft = 2 * N
    f_hat = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
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

def local_refinement(h, start_time, max_time):
    """Local refinement using FFT-based sensitivity peak reduction with adaptive
    threshold, smooth peak mask, and backtracking line search."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.03
    threshold_frac = 0.97
    n_fft = 2 * N
    
    no_improve_count = 0
    max_no_improve = 15
    
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
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        threshold = max_c * threshold_frac
        lower_thresh = max_c * 0.95
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        ch_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
        
        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std
        
        best_alpha = alpha
        best_h_candidate = None
        best_val_candidate = current_val
        
        for trial_alpha in [alpha, alpha * 0.5, alpha * 2.0, alpha * 0.25, alpha * 4.0]:
            h_new = current_h * (1.0 - trial_alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            h_new = h_new * (N / s_new)
            
            f_new_hat = np.fft.rfft(h_new, n=n_fft)
            conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < best_val_candidate:
                best_val_candidate = val_new
                best_alpha = trial_alpha
                best_h_candidate = h_new
        
        if best_h_candidate is not None and best_val_candidate < current_val:
            current_h = best_h_candidate
            if best_alpha > alpha:
                alpha = min(alpha * 1.1, 0.2)
            else:
                alpha = max(alpha * 0.9, 0.01)
        else:
            alpha *= 0.7
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def spectral_phase_refinement(h, start_time, max_time):
    """Spectral phase optimization: represent f as |FFT^{-1}(A * e^{i*phase})| and
    optimize phase angles to flatten autoconvolution peak.
    
    Key insight: In frequency domain, autoconvolution magnitude is |F|^2 where F = FFT(f).
    By parameterizing f = |IFFT(magnitude * exp(i*phase))|, we can adjust phases to
    reduce time-domain peak while maintaining non-negativity through the absolute value.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    n_fft = 2 * N
    n_freq = n_fft // 2 + 1  # Number of unique frequency bins for real signal
    
    # Initialize phases from current signal
    def get_phases(f):
        f_hat = np.fft.rfft(f, n=n_fft)
        magnitudes = np.abs(f_hat)
        phases = np.angle(f_hat)
        return magnitudes, phases
    
    # Reconstruct time-domain signal from magnitudes and phases
    def reconstruct(magnitudes, phases):
        f_hat = magnitudes * np.exp(1j * phases)
        f = np.fft.irfft(f_hat, n=n_fft)[:N]
        return np.maximum(f, 0.0)
    
    # Initial magnitudes and phases
    magnitudes, phases = get_phases(current_h)
    
    # Save DC component (phase 0) to preserve normalization
    dc_magnitude = magnitudes[0].copy()
    
    alpha = 0.1
    no_improve = 0
    max_no_improve = 25
    
    while time.time() - start_time < max_time:
        # Reconstruct current signal
        current_h = reconstruct(magnitudes, phases)
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        # Recompute magnitudes and phases from normalized signal
        magnitudes, phases = get_phases(current_h)
        
        conv = fft_convolve(current_h)
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
        
        # Compute gradient of peak autoconvolution w.r.t. phases
        # d/dphase[k] max(conv) ≈ sum over peaks of d(conv[peak])/dphase[k]
        # conv = |F|^2 in freq domain, but we need time-domain peak
        # Use sensitivity approach: gradient of smooth max w.r.t. phases
        
        # Compute peak mask in time domain
        threshold = max_c * 0.98
        lower_thresh = max_c * 0.95
        peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
        
        # For phase optimization, we need: d(conv)/dphase
        # conv = IFFT(|F|^2) where F = mag * exp(i*phase)
        # This is complex; use finite difference or analytical gradient
        
        # Analytical approach: 
        # f = IFFT(mag * exp(i*phase))
        # conv = f * f (autoconvolution)
        # d(conv)/dphase[k] = 2 * Re{IFFT(2*F * conj(dF/dphase[k]))}
        # dF/dphase[k] = i * F * (delta[k] - mean(delta)) for phase perturbation
        
        # Simpler: use subgradient of smooth max
        # d smooth_max(conv, beta) / dphase ≈ (1/beta) * sum_j exp(beta*conv[j]/max) * d(conv[j])/dphase
        
        # For efficiency, compute gradient via FFT
        # d(conv[j])/dphase[k] = 2 * Re{IFFT(2 * F * conj(i * F * e_k))}[j]
        # where e_k is unit vector at frequency k
        
        # This gives: d(conv)/dphase = 2 * Re{IFFT(2 * F * conj(i * F)) * e_k}
        # = 2 * Re{IFFT(-2i * |F|^2) * e_k}
        # = 2 * Re{(-2i) * IFFT(|F|^2) * e_k}  [since IFFT is linear]
        
        # Actually, let's use a more direct approach with finite differences
        # for a subset of frequencies to keep it fast
        
        # Select a random subset of frequency indices to update
        n_update = min(50, n_freq - 1)
        update_indices = np.random.choice(1, n_freq, size=n_update, replace=False)
        
        # Compute gradient for these indices using finite difference
        eps = 0.01
        grad_phases = np.zeros(n_freq)
        
        for idx in update_indices:
            # Forward difference
            phases_plus = phases.copy()
            phases_plus[idx] += eps
            h_plus = reconstruct(magnitudes, phases_plus)
            s_plus = np.sum(h_plus)
            if s_plus < 1e-12:
                continue
            h_plus = h_plus * (N / s_plus)
            conv_plus = fft_convolve(h_plus)
            val_plus = np.max(conv_plus) / (N * N)
            
            # Backward difference
            phases_minus = phases.copy()
            phases_minus[idx] -= eps
            h_minus = reconstruct(magnitudes, phases_minus)
            s_minus = np.sum(h_minus)
            if s_minus < 1e-12:
                continue
            h_minus = h_minus * (N / s_minus)
            conv_minus = fft_convolve(h_minus)
            val_minus = np.max(conv_minus) / (N * N)
            
            grad_phases[idx] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient descent step on phases
        grad_norm = np.linalg.norm(grad_phases[update_indices])
        if grad_norm > 1e-15:
            grad_phases[update_indices] /= grad_norm
            
            # Try multiple step sizes
            best_alpha = alpha
            best_candidate = None
            best_val_candidate = current_val
            
            for trial_alpha in [alpha, alpha * 0.5, alpha * 2.0, alpha * 0.25]:
                phases_new = phases.copy()
                phases_new[update_indices] -= trial_alpha * grad_phases[update_indices]
                
                h_new = reconstruct(magnitudes, phases_new)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                conv_new = fft_convolve(h_new)
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val_candidate:
                    best_val_candidate = val_new
                    best_alpha = trial_alpha
                    best_candidate = (h_new, phases_new)
            
            if best_candidate is not None and best_val_candidate < current_val:
                current_h, phases = best_candidate
                # Update magnitudes from new signal
                magnitudes, _ = get_phases(current_h)
                if best_alpha > alpha:
                    alpha = min(alpha * 1.1, 0.5)
                else:
                    alpha = max(alpha * 0.9, 0.01)
            else:
                alpha *= 0.7
                if alpha < 1e-6:
                    break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric initialization with cosine bumps
    # Reserve time: 25% for parametric, 40% for local refinement, 25% for wavelet, 10% for polish
    param_budget_end = start_time + (TIME_LIMIT - 5) * 0.25
    
    for n_c in [2, 3, 4]:
        if time.time() > param_budget_end:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
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
                        options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7}
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
    
    # Phase 2: FFT-based local refinement
    if time.time() < TIME_LIMIT - 55:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 55
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 3: Spectral phase optimization
    if time.time() < TIME_LIMIT - 20:
        spectral_f, spectral_score = spectral_phase_refinement(
            best_f, start_time, TIME_LIMIT - 20
        )
        if spectral_score < best_score:
            best_f = spectral_f
            best_score = spectral_score
    
    # Phase 4: Final high-precision polish
    if time.time() < TIME_LIMIT - 2:
        best_f = best_f.copy()
        alpha_small = 0.005
        n_fft = 2 * N
        
        for _ in range(200):
            if time.time() > TIME_LIMIT - 2:
                break
            s = np.sum(best_f)
            if s < 1e-12:
                break
            best_f = best_f * (N / s)
            f_hat = np.fft.rfft(best_f, n=n_fft)
            conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2*N-1]
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            threshold = max_c * 0.998
            lower_thresh = max_c * 0.995
            peak_mask = np.clip((conv - lower_thresh) / (threshold - lower_thresh + 1e-15), 0, 1)
            
            pm_hat = np.fft.rfft(peak_mask, n=n_fft)
            ch_hat = np.fft.rfft(best_f, n=n_fft)
            sens = np.fft.irfft(pm_hat * np.conj(ch_hat), n=n_fft)[:N]
            
            sens_mean = np.mean(sens)
            sens_std = np.std(sens) + 1e-12
            sens_norm = (sens - sens_mean) / sens_std
            
            best_alpha = alpha_small
            best_candidate = None
            best_val = current_val
            
            for trial_alpha in [alpha_small, alpha_small * 0.5, alpha_small * 2.0]:
                h_new = best_f * (1.0 - trial_alpha * sens_norm)
                h_new = np.maximum(h_new, 0.0)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                f_new_hat = np.fft.rfft(h_new, n=n_fft)
                conv_new = np.fft.irfft(f_new_hat * f_new_hat, n=n_fft)[:2*N-1]
                val_new = np.max(conv_new) / (N * N)
                
                if val_new < best_val:
                    best_val = val_new
                    best_alpha = trial_alpha
                    best_candidate = h_new
            
            if best_candidate is not None and best_val < current_val:
                best_f = best_candidate
                alpha_small = min(alpha_small * 1.05, 0.02)
            else:
                alpha_small *= 0.8
                if alpha_small < 1e-7:
                    break
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
