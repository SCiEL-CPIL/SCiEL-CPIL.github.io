"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

New Strategy:
1. Work in the frequency domain (Fourier coefficients) instead of time domain.
2. Use a spectral parameterization: the signal is the inverse FFT of nonnegative
   frequency amplitudes with fixed phase 0 (real, even, nonnegative signal).
3. Optimize the amplitudes using projected gradient descent on the smooth max
   of the autoconvolution, with projection onto the nonnegative orthant and
   normalization constraints.
4. This directly optimizes the second-order structure (power spectrum) that
   determines the autoconvolution peak.
5. Add a time-domain polish pass at the end.
"""

import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
TIME_LIMIT = 135

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

def spectral_to_time(amps):
    """Convert nonnegative frequency amplitudes to time-domain signal.
    
    We use a real, even signal: f(t) = sum_k amps[k] * cos(2*pi*k*t/N)
    This guarantees f is real and even (symmetric about 0).
    We take amps as the amplitude of each cosine component.
    """
    n_freq = len(amps)
    # Build the frequency-domain representation
    # For a real even signal of length N, FFT has N/2+1 unique frequencies
    # f[n] = (1/N) * sum_k amps[k] * exp(2j*pi*k*n/N) + conjugate terms
    # For even real: f[n] = (2/N) * sum_{k=1}^{N/2-1} amps[k] * cos(2*pi*k*n/N) 
    #                + (1/N) * amps[0] + (1/N) * amps[N/2] * cos(pi*n)
    
    # Simpler: use IFFT directly
    # Create full FFT vector
    fft_vec = np.zeros(N, dtype=np.complex128)
    fft_vec[0] = amps[0]
    for k in range(1, n_freq):
        if k < N // 2:
            fft_vec[k] = amps[k]
            fft_vec[N - k] = amps[k]
        elif k == N // 2:
            fft_vec[k] = amps[k]
    
    f = np.fft.ifft(fft_vec).real
    return f

def time_to_spectral(f):
    """Extract nonnegative amplitudes from time-domain signal."""
    fft_f = np.fft.fft(f)
    # Take magnitudes of first N/2+1 bins
    n_freq = N // 2 + 1
    amps = np.abs(fft_f[:n_freq])
    return amps

def spectral_objective_and_grad(amps, beta=20.0):
    """Objective and gradient for spectral optimization.
    
    We minimize max(conv(f,f)) / (sum(f))^2 where f = IFFT(amps)
    
    The autoconvolution in frequency domain: conv = IFFT(FFT(f) * FFT(f))
    Since FFT(f) = amps (by construction), conv = IFFT(amps * amps)
    
    Actually, let's be more careful:
    f = IFFT(fft_vec) where fft_vec is built from amps
    FFT(f) = fft_vec (up to scaling)
    conv = IFFT(FFT(f) * FFT(f)) = IFFT(fft_vec * fft_vec)
    
    But we need to handle the normalization properly.
    """
    n_freq = len(amps)
    
    # Build full FFT vector
    fft_vec = np.zeros(N, dtype=np.complex128)
    fft_vec[0] = amps[0]
    for k in range(1, n_freq):
        if k < N // 2:
            fft_vec[k] = amps[k]
            fft_vec[N - k] = amps[k]
        elif k == N // 2:
            fft_vec[k] = amps[k]
    
    # Time-domain signal
    f = np.fft.ifft(fft_vec).real
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(amps)
    
    # Autoconvolution: conv = IFFT(FFT(f) * FFT(f))
    # But FFT(f) is not exactly fft_vec because we clipped f to nonnegative
    # So we need to recompute
    fft_f = np.fft.fft(f, n=2*N)
    conv = np.fft.ifft(fft_f * fft_f).real[:2*N-1]
    
    # Smooth max
    x_max = np.max(conv)
    if x_max < 1e-15:
        return smooth_max(conv, beta) / (s * s), np.zeros_like(amps)
    
    shifted = conv - x_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    p = exp_shifted / sum_exp
    
    smooth_max_val = x_max + np.log(sum_exp) / beta
    val = smooth_max_val / (s * s)
    
    # Gradient: d/damps = (1/s^2) * d(smooth_max)/damps - (2/s^3) * smooth_max * d(s)/damps
    # d(smooth_max)/damps = sum_i p_i * d(conv_i)/damps
    # d(conv_i)/damps = d/damps [IFFT(FFT(f)^2)_i]
    # This is complex; use finite differences or chain rule
    
    # Chain rule: conv = IFFT(FFT(f)^2)
    # d(conv)/d(f_j) = 2 * IFFT(FFT(f) * E_j) where E_j has 1 at position j
    # d(f)/d(amps_k) = d/d(amps_k) IFFT(fft_vec)
    
    # For efficiency, compute gradient via FFT
    # d(f)/d(amps_k): 
    #   For k=0: f = IFFT(fft_vec), d(f)/d(amps_0) = IFFT(e_0) where e_0 has 1 at index 0
    #   For k>0, k<N/2: d(f)/d(amps_k) = IFFT(e_k + e_{N-k})
    #   For k=N/2: d(f)/d(amps_k) = IFFT(e_k)
    
    # grad_conv_k = 2 * IFFT(FFT(f) * FFT(d(f)/d(amps_k)))
    # But FFT(d(f)/d(amps_k)) is just the basis vector in frequency domain
    
    # Let's compute:
    # g_k = d(f)/d(amps_k)
    # grad_smooth_max_k = sum(p * 2 * IFFT(fft_f * FFT(g_k)))
    # FFT(g_k) for k=0: vector with 1 at index 0, 0 elsewhere -> just [1, 0, ..., 0]
    # For k in (0, N/2): FFT(g_k) has 1 at index k and 1 at index N-k
    # For k=N/2: FFT(g_k) has 1 at index N/2
    
    # So: grad_smooth_max_k = 2 * sum(p * IFFT(fft_f * basis_k))
    # where basis_k is the appropriate basis in frequency domain
    
    # IFFT(fft_f * basis_k) = (1/(2N)) * fft_f[k] * exp(2j*pi*k*n/(2N)) summed... 
    # Actually, IFFT(fft_f * e_k) where e_k has 1 at position k:
    # = (1/(2N)) * sum_m fft_f[m] * e_k[m] * exp(2j*pi*m*n/(2N))
    # = (1/(2N)) * fft_f[k] * exp(2j*pi*k*n/(2N))
    
    # This is getting complex. Let's use a simpler approach:
    # Compute gradient numerically with small perturbations for a few key dimensions
    # Or use the fact that we can compute the gradient of the objective w.r.t. f
    # and then chain through f = IFFT(fft_vec)
    
    # Gradient of val w.r.t. f:
    # d(val)/d(f_j) = (1/s^2) * d(smooth_max)/d(f_j) - (2/s^3)*smooth_max * 1
    # d(smooth_max)/d(f_j) = sum_i p_i * d(conv_i)/d(f_j)
    # d(conv_i)/d(f_j) = 2 * (f * delta_j)_i where delta_j is unit vector at j
    # = 2 * f[i-j] (circular, but we use full conv so it's linear)
    
    # For full convolution: conv[i] = sum_j f[j]*f[i-j]
    # d(conv[i])/d(f[j]) = f[i-j] + f[j-i] (but for nonneg and our setup, 
    #   actually conv[i] = sum_k f[k]*f[i-k], so d/d(f[j]) = f[i-j] + f[j-(i-j)]... 
    #   Let's just say d(conv[i])/d(f[j]) = f[i-j] + f[j-i] for the linear conv)
    
    # Actually for autoconvolution: conv[i] = sum_j f[j] * f[i-j]
    # d(conv[i])/d(f[j]) = f[i-j] + f[j-i]  (since f appears in both factors)
    # Wait, that's not right. conv[i] = sum_k f[k] f[i-k]
    # d/d(f[j]) = f[i-j] (from first factor when k=j) + f[j] (from second factor when i-k=j, i.e., k=i-j)
    # So d(conv[i])/d(f[j]) = f[i-j] + f[i-j] = 2*f[i-j] if we consider the sum properly
    # Actually: sum_k f[k] f[i-k], derivative w.r.t. f[j]:
    #   term k=j: f[j] * f[i-j] -> derivative is f[i-j]
    #   term k=i-j: f[i-j] * f[j] -> derivative is f[i-j]  
    #   Wait, these are the same term if j = i-j, i.e., i=2j
    #   More carefully: d/d(f[j]) sum_k f[k] f[i-k] = sum_k [delta(k,j) f[i-k] + f[k] delta(i-k,j)]
    #   = f[i-j] + f[i-j] = 2 f[i-j]  (when both terms exist, which they do for linear conv)
    #   But if j > i or i-j < 0, some terms don't exist. For full conv with proper bounds:
    #   conv[i] = sum_{k=max(0,i-(N-1))}^{min(i,N-1)} f[k] f[i-k]
    #   d/d(f[j]) = f[i-j] (if 0<=i-j<N) + f[i-j] (if 0<=i-j<N) = 2*f[i-j] (if 0<=i-j<N)
    
    # So d(smooth_max)/d(f[j]) = sum_i p_i * 2 * f[i-j] * I(0<=i-j<N)
    #                          = 2 * sum_{m} p_{m+j} * f[m] * I(0<=m<N, 0<=m+j<2N-1)
    #                          = 2 * (p * f)_j  (cross-correlation)
    
    # We can compute this via FFT:
    # cross_corr[j] = sum_m p[m+j] * f[m]
    # This is like a correlation. Let's use FFT.
    
    fft_p = np.fft.fft(p, n=2*N)
    fft_f_full = np.fft.fft(f, n=2*N)
    # cross_corr[j] = sum_m p[m+j] f[m]
    # = sum_m p[j+m] f[m]
    # This is (p * f) at lag j, which is IFFT(conj(FFT(p)) * FFT(f)) shifted
    # Actually: sum_m p[m+j] f[m] = sum_m p[j+m] f[m]
    # Let n = m+j, then m = n-j: sum_n p[n] f[n-j]
    # This is the cross-correlation of p and f at lag j.
    # cross_corr = IFFT(conj(FFT(p)) * FFT(f)) gives sum_n p[n] conj(f[n-j]) for real f: sum_n p[n] f[n-j]
    # Hmm, let's just use np.correlate or direct FFT
    
    # Simpler: use the fact that sum_m p[m+j] f[m] = (p ⊛ f)[j] where ⊛ is cross-correlation
    # cross_corr = IFFT(FFT(p) * conj(FFT(f)))  for circular correlation
    # For linear, we need zero-padding
    
    fft_p_pad = np.fft.fft(p, n=4*N)
    fft_f_pad = np.fft.fft(f, n=4*N)
    cross_corr = np.fft.ifft(fft_p_pad * np.conj(fft_f_pad)).real[:2*N-1]
    
    # d(smooth_max)/d(f[j]) = 2 * cross_corr[j] for j in range(N)
    # But we need to be careful about the bounds. For our purpose, approximate:
    grad_smooth_max_f = 2.0 * cross_corr[:N]
    
    # d(s)/d(f[j]) = 1
    grad_s_f = np.ones(N)
    
    # d(val)/d(f[j]) = (1/s^2) * grad_smooth_max_f[j] - (2/s^3) * smooth_max_val * grad_s_f[j]
    grad_val_f = (grad_smooth_max_f / (s * s)) - (2.0 * smooth_max_val / (s * s * s))
    
    # Now chain through f = IFFT(fft_vec)
    # f[n] = (1/N) * sum_k fft_vec[k] * exp(2j*pi*k*n/N)
    # d(f[n])/d(amps_k):
    #   For k=0: d(f[n])/d(amps_0) = (1/N) * 1 = 1/N
    #   For 0<k<N/2: d(f[n])/d(amps_k) = (1/N) * (exp(2j*pi*k*n/N) + exp(-2j*pi*k*n/N)) = (2/N) * cos(2*pi*k*n/N)
    #   For k=N/2: d(f[n])/d(amps_k) = (1/N) * exp(2j*pi*(N/2)*n/N) = (1/N) * (-1)^n
    
    # grad_val_amps[k] = sum_n grad_val_f[n] * d(f[n])/d(amps_k)
    grad_val_amps = np.zeros(n_freq)
    
    # k=0
    grad_val_amps[0] = np.sum(grad_val_f) / N
    
    # 0 < k < N/2
    for k in range(1, N // 2):
        if k >= n_freq:
            break
        cos_term = (2.0 / N) * np.cos(2.0 * np.pi * k * np.arange(N) / N)
        grad_val_amps[k] = np.dot(grad_val_f, cos_term)
    
    # k = N/2
    if n_freq > N // 2:
        grad_val_amps[N // 2] = np.sum(grad_val_f * ((-1) ** np.arange(N))) / N
    
    return val, grad_val_amps

def local_refinement(h, start_time, max_time, alpha_init=0.1, threshold_frac=0.9):
    """Local refinement using FFT-based weighted sensitivity and adaptive step sizes."""
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
        
        threshold = max_c * threshold_frac
        peak_mask = np.zeros(len(conv), dtype=np.float64)
        above = conv >= threshold
        if np.any(above):
            denom = max_c - threshold
            if denom > 1e-15:
                peak_mask[above] = (conv[above] - threshold) / denom
            else:
                peak_mask[above] = 1.0
        
        fft_h = np.fft.fft(current_h, n=2*N)
        fft_pm = np.fft.fft(peak_mask, n=2*N)
        cross_corr = np.fft.ifft(fft_h * np.conj(fft_pm)).real[:N]
        sensitivity = cross_corr
        
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
            alpha = min(alpha * 1.04, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Spectral optimization
    n_freq = N // 2 + 1
    
    # Initialize amplitudes from a smooth kernel (Fejer-like)
    # Fejer kernel in frequency domain: triangular spectrum
    k = np.arange(n_freq)
    amps_init = np.maximum(1.0 - k / (N // 2), 0.0)
    amps_init[0] = 1.0
    
    # Try multiple restarts with different initializations
    for restart in range(8):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        if restart == 0:
            amps = amps_init.copy()
        elif restart == 1:
            # Broad spectrum
            amps = np.ones(n_freq) * 0.5
            amps[0] = 1.0
        elif restart == 2:
            # Narrow spectrum
            amps = np.zeros(n_freq)
            amps[:n_freq//4] = 1.0
            amps[0] = 1.5
        elif restart == 3:
            # Random smooth
            rng = np.random.RandomState(42 + restart)
            amps = rng.uniform(0.1, 1.0, size=n_freq)
            # Smooth it
            from scipy.ndimage import gaussian_filter1d
            amps = gaussian_filter1d(amps, sigma=5)
            amps[0] = 1.0
        elif restart == 4:
            # Cosine decay
            amps = 0.5 * (1 + np.cos(np.pi * k / (N // 2)))
            amps[0] = 1.0
        elif restart == 5:
            # Two-bump spectrum
            centers = [N//8, 3*N//8]
            widths = [N//16, N//12]
            amps = np.zeros(n_freq)
            for c, w in zip(centers, widths):
                amps += np.exp(-0.5 * ((k - c) / w) ** 2)
            amps[0] = 1.0
        else:
            # Random with smoothing
            rng = np.random.RandomState(100 + restart)
            amps = rng.uniform(0.0, 1.0, size=n_freq)
            from scipy.ndimage import gaussian_filter1d
            amps = gaussian_filter1d(amps, sigma=10)
            amps[0] = 1.0
        
        # Ensure nonnegative
        amps = np.maximum(amps, 0.0)
        
        # Gradient descent with projection
        lr = 0.01
        beta = 15.0
        
        for step in range(3000):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            val, grad = spectral_objective_and_grad(amps, beta)
            
            if not np.isfinite(val):
                break
            
            # Gradient step
            amps_new = amps - lr * grad
            
            # Project onto nonnegative orthant
            amps_new = np.maximum(amps_new, 0.0)
            
            # Optional: normalize to keep scale reasonable
            # The objective is scale-invariant (val = max(conv)/s^2), so scale doesn't matter
            # But we need amps[0] > 0 to ensure s > 0
            if amps_new[0] < 1e-6:
                amps_new[0] = 1e-6
            
            # Check if improved
            f_new = spectral_to_time(amps_new)
            f_new = np.maximum(f_new, 0.0)
            score_new = compute_score(f_new)
            
            if score_new < best_score:
                best_score = score_new
                best_f = f_new.copy()
            
            # Adaptive learning rate
            if score_new < val:  # improved
                lr = min(lr * 1.02, 0.05)
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break
            
            amps = amps_new
        
        # Increase beta for sharper max approximation in subsequent restarts
        beta = min(beta * 1.5, 50.0)
    
    # Local refinement in time domain
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
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
