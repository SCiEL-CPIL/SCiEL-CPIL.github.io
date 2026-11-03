"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Use cosine bump parametric family for initialization, then refine
with spectral coordinate descent in FFT domain. The spectral approach adjusts
frequency components to flatten the autoconvolution peak, providing a
different optimization path than time-domain methods.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

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
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.fft(f, size)
    conv = np.fft.ifft(F * F).real[:2 * n - 1]
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
    return smooth_max(conv, beta) / (s * s)

def spectral_refinement(h, start_time, max_time):
    """Spectral coordinate descent: adjust FFT components to reduce peak autoconvolution.
    
    The autoconvolution in frequency domain is |F|^2. We adjust each frequency
    component's magnitude to flatten the time-domain peak while maintaining
    non-negativity of the time-domain signal.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.1
    patience = 0
    max_patience = 200
    step_count = 0
    
    # FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize to sum = N for stable computation
        current_h = current_h * (N / s)
        
        # Compute FFT
        F = np.fft.fft(current_h, fft_size)
        
        # Compute autoconvolution
        conv = np.fft.ifft(F * F).real[:2*N-1]
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
        
        # Find the peak lag
        k_star = np.argmax(conv)
        
        # For each frequency bin, compute how much it contributes to the peak
        # conv[k] = (1/fft_size) * sum_m F[m] * F[k-m] (circular)
        # For linear conv, we need to be more careful, but as an approximation:
        # The derivative of conv[k_star] w.r.t. F[m] is proportional to F[k_star - m]
        
        # Compute gradient in frequency domain
        # d(conv[k_star])/d(F[m]) ~ F[(k_star - m) % fft_size]
        m_indices = np.arange(fft_size)
        grad_F = np.zeros(fft_size, dtype=complex)
        for m in range(fft_size // 2 + 1):
            # For real signals, F[fft_size-m] = conj(F[m])
            # d(conv[k])/d(F[m]) = F[k-m] for the circular case
            idx = (k_star - m) % fft_size
            grad_F[m] = F[idx]
            if m > 0 and m < fft_size // 2:
                grad_F[fft_size - m] = np.conj(F[(k_star + m) % fft_size])
        
        # We want to reduce |F[m]| where grad_F[m] * conj(F[m]) is large (constructive at peak)
        # and increase where it's negative (destructive at peak)
        # This is equivalent to: F_new[m] = F[m] - alpha * grad_F[m] * conj(F[m]) / |F[m]|
        
        # Simpler approach: adjust magnitude based on correlation with peak
        # Compute the contribution of each frequency to the peak
        F_abs = np.abs(F)
        F_phase = np.angle(F)
        
        # For each frequency, compute the phase that would minimize the peak
        # This is complex, so use a simpler heuristic:
        # Reduce magnitudes of frequencies that have high power and contribute to peak
        
        # Compute power spectrum
        power = F_abs ** 2
        power_mean = np.mean(power)
        power_std = np.std(power) + 1e-12
        
        # Identify high-power frequencies
        high_power_mask = power > power_mean + 0.5 * power_std
        
        # For high-power frequencies, slightly reduce their magnitude
        # This tends to flatten the peak
        F_new = F.copy()
        for m in range(fft_size // 2 + 1):
            if high_power_mask[m]:
                F_new[m] *= (1.0 - alpha * 0.1)
                if m > 0 and m < fft_size // 2:
                    F_new[fft_size - m] *= (1.0 - alpha * 0.1)
        
        # Inverse FFT to get new time-domain signal
        h_new = np.fft.ifft(F_new).real[:N]
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        # Check improvement
        val_new = compute_score(h_new)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.5)
            step_count += 1
        else:
            alpha *= 0.95
            if alpha < 1e-6:
                break
    
    return best_h, best_val

def time_domain_refinement(h, start_time, max_time):
    """Time-domain coordinate descent with exact 1D minimization."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        
        # Normalize
        current_h = current_h * (N / s)
        
        # Compute autoconvolution
        conv = fft_autoconv(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find the peak lag
        k_star = np.argmax(conv)
        
        # For each bin j, compute the contribution to the peak
        # conv[k_star] = sum_j f[j] * f[k_star - j]
        # d(conv[k_star])/d(f[j]) = f[k_star - j]
        
        # Try adjusting one bin at a time
        improved = False
        for j in range(N):
            if time.time() - start_time > max_time:
                break
            
            # Compute the optimal value for f[j] that minimizes conv[k_star]
            # conv[k_star] = f[j] * f[k_star - j] + sum_{j' != j} f[j'] * f[k_star - j']
            # Let A = sum_{j' != j} f[j'] * f[k_star - j']
            # Let B = f[k_star - j]
            # conv[k_star] = f[j] * B + A
            # To minimize, we want f[j] as small as possible (since B >= 0)
            # But we need to maintain non-negativity and constant sum
            
            # Simple approach: reduce f[j] slightly and redistribute
            if current_h[j] > 1e-10:
                delta = current_h[j] * 0.1
                h_new = current_h.copy()
                h_new[j] -= delta
                # Redistribute to other bins
                other_bins = np.delete(np.arange(N), j)
                h_new[other_bins] += delta / len(other_bins)
                h_new = np.maximum(h_new, 0.0)
                
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new * (N / s_new)
                
                val_new = compute_score(h_new)
                if val_new < current_val:
                    current_h = h_new
                    current_val = val_new
                    improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try different numbers of cosine bump components
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        rng = np.random.RandomState(42 + n_c)
        best_params = None
        best_obj = np.inf
        
        # Annealed beta: start lower, increase for sharper max approximation
        betas = [5.0, 10.0, 20.0]
        
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            for restart in range(6):
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
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
    
    # Spectral refinement: adjust FFT components to reduce peak
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Time-domain refinement: coordinate descent
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = time_domain_refinement(
            best_f, start_time, TIME_LIMIT - 2
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
