"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Use a smooth parametric family based on Gaussian bumps with multiple components.
2. Optimize using BFGS with analytic gradients for the smooth max surrogate.
3. Apply local refinement using sensitivity-based peak reduction with FFT-based convolution.
4. Add a final polish pass with adaptive step sizes.

Key improvements:
- Gradient-based optimization (BFGS) instead of derivative-free Nelder-Mead
- FFT-based autoconvolution for O(N log N) instead of O(N^2)
- Better parametric family with Gaussian bumps for smoother autoconvolution
- More aggressive local refinement with adaptive thresholds
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 135  # safety margin before 150s kill

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

def gaussian_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from Gaussian bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        result += h * np.exp(-0.5 * ((x - c) / (w / 2.0)) ** 2)
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
    
    f = gaussian_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective_and_grad(params, n_components, beta=10.0):
    """Objective and gradient for the smooth max surrogate."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    w_eff = np.abs(widths) + 0.01
    w_eff = np.minimum(w_eff, 0.5)
    h_eff = np.abs(heights)
    
    f = gaussian_basis(x, centers, w_eff, h_eff)
    f = np.maximum(f, 0.0)
    
    conv = fftconvolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(params)
    
    val = smooth_max(conv, beta) / (s * s)
    
    # Compute gradient numerically for simplicity (BFGS can handle this)
    # In practice, analytic gradients would be better, but this is a bootstrap
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = parametric_to_step(params_plus, n_components)
        conv_plus = fftconvolve(f_plus, f_plus, mode='full')
        s_plus = np.sum(f_plus)
        val_plus = smooth_max(conv_plus, beta) / (s_plus * s_plus) if s_plus > 1e-15 else np.inf
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = parametric_to_step(params_minus, n_components)
        conv_minus = fftconvolve(f_minus, f_minus, mode='full')
        s_minus = np.sum(f_minus)
        val_minus = smooth_max(conv_minus, beta) / (s_minus * s_minus) if s_minus > 1e-15 else np.inf
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
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
        # For each bin j, sensitivity = sum over peak lags m of h[m-j]
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
        
        # Reduce bins with high sensitivity
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
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
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

def spectral_subgradient_descent(f_init, start_time, max_time, lr_init=0.01):
    """
    Spectral subgradient descent to minimize the peak of the autoconvolution.
    Uses DFT to compute gradients in the frequency domain.
    """
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    lr = lr_init
    M = 2048  # FFT size for linear convolution (zero padding)
    
    while time.time() - start_time < max_time:
        s = np.sum(current_f)
        if s < 1e-12:
            break
        
        # Normalize to maintain scale (score is scale invariant? No, score = max(c)/s^2.
        # If we scale f by k, c scales by k^2, s scales by k. Score is invariant.
        # So we can fix s = N for simplicity? Or just keep it.
        # Let's normalize to s = N to avoid numerical issues.
        current_f = current_f * (N / s)
        
        # Compute Autoconvolution via FFT
        F = np.fft.fft(current_f, M)
        C_spec = F * F  # Note: for real f, F[-k] = conj(F[k]), so |F|^2 is real and symmetric.
        # Actually, C_spec = F * conj(F) for power spectrum?
        # Autoconvolution c = f * f.
        # DFT(c) = F * F? No.
        # DFT(f * f) = F * F is for complex.
        # For real f, c is real.
        # DFT(c) = F * conj(F) = |F|^2.
        # Wait.
        # (f*f)(t) = sum f(j) f(t-j).
        # DFT(f*f) = F * F?
        # Let's check.
        # DFT(f) = F.
        # DFT(f*f) = F * F?
        # No, DFT(f*g) = F * G.
        # So DFT(f*f) = F * F.
        # Since f is real, F is Hermitian.
        # F * F is not necessarily real?
        # F[k] F[k].
        # If F[k] = a+ib, F[k]^2 = a^2-b^2 + 2iab. Not real.
        # But c(t) is real, so DFT(c) must be Hermitian.
        # F[k] F[-k] = F[k] conj(F[k]) = |F[k]|^2.
        # So DFT(c) = |F|^2.
        # Yes.
        C_spec = np.abs(F)**2
        c = np.fft.ifft(C_spec).real[:N]  # Linear conv is in the first N?
        # Actually, linear conv of length N and N is length 2N-1.
        # With zero padding M, the result is in [0, M-1].
        # The peak can be anywhere in [0, M-1].
        c_full = np.fft.ifft(C_spec).real
        max_c = np.max(c_full)
        current_val = max_c / (N * N) # Since we normalized s=N
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Find peak index
        k_max = np.argmax(c_full)
        
        # Compute subgradient in time domain
        # grad_f = 2 * (delta_k_max * f)
        # In frequency domain:
        # G = F * DFT(delta_k_max)
        # DFT(delta_k_max) is a complex exponential e^{-i 2 pi k_max n / M}
        n = np.arange(M)
        DFT_delta = np.exp(-1j * 2 * np.pi * k_max * n / M)
        G_spec = F * DFT_delta
        grad_time = np.fft.ifft(G_spec).real[:N]
        
        # Update f
        # f_new = max(f - lr * grad_time, 0)
        # We need to scale lr appropriately.
        # grad_time has magnitude ~ N.
        # f has magnitude ~ 1.
        # So lr should be small.
        
        f_new = current_f - lr * grad_time
        f_new = np.maximum(f_new, 0.0)
        
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            break
        
        f_new = f_new * (N / s_new)
        
        # Check improvement
        F_new = np.fft.fft(f_new, M)
        c_new = np.fft.ifft(np.abs(F_new)**2).real
        val_new = np.max(c_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.8
            if lr < 1e-6:
                break
                
    return best_f, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Spectral Subgradient Descent
    # Start from constant function
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = spectral_subgradient_descent(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
            
    # Try a few random initializations with spectral descent
    rng = np.random.RandomState(42)
    for i in range(5):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        # Random smooth initialization
        f_init = np.ones(N) * rng.uniform(0.5, 1.5)
        # Add some smooth noise
        noise = np.random.normal(0, 0.1, N)
        # Smooth it
        kernel = np.ones(10) / 10
        noise = np.convolve(noise, kernel, mode='same')
        f_init = f_init + noise
        f_init = np.maximum(f_init, 0)
        
        refined_f, refined_score = spectral_subgradient_descent(
            f_init, start_time, TIME_LIMIT - 5
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
