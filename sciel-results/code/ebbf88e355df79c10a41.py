"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Spectral-domain optimization: parameterize the function by its Fourier coefficients.
2. The autoconvolution is the inverse FFT of |F|^2. Minimizing the peak of the autoconvolution
   is equivalent to minimizing the peak of the inverse FFT of |F|^2.
3. We optimize the Fourier coefficients directly, projecting back to the time domain
   to enforce non-negativity.
4. Use a smooth max surrogate for the peak.
5. Use L-BFGS-B from scipy for efficient optimization in the spectral domain.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
DT = 0.5 / N  # interval width
TIME_LIMIT = 140  # safety margin before 150s kill


def autoconvolve_fft(f):
    """Compute autoconvolution using FFT."""
    n = 2 * N
    F = np.fft.rfft(f, n=n)
    conv = np.fft.irfft(F * F, n=n)[:2 * N - 1]
    return conv


def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = autoconvolve_fft(f)
    return np.max(conv) / (s * s)


def time_to_freq(f):
    """Convert time-domain signal to frequency-domain (real FFT)."""
    n = 2 * N
    return np.fft.rfft(f, n=n)


def freq_to_time(F):
    """Convert frequency-domain (real FFT) back to time-domain."""
    n = 2 * N
    return np.fft.irfft(F, n=n)[:N]


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


def objective_freq(params, beta):
    """
    Objective function in the frequency domain.
    params: real and imaginary parts of the FFT coefficients (interleaved or separate).
    We'll use a representation where params = [Re(F0), Im(F1), Re(F1), Im(F2), ...]
    but since F0 is real (DC component) and F[n] = conj(F[N-n]) for real f,
    we only need to optimize the first N+1 coefficients.
    
    Actually, for real f, the FFT has conjugate symmetry: F[k] = conj(F[N-k]).
    So we only need to store F[0] (real) and F[1..N-1] (complex), and F[N] (real if N even).
    For N=600 (even), we have F[0] (real), F[1..299] (complex), F[300] (real).
    Total parameters: 1 + 2*299 + 1 = 600.
    
    Let's use a simpler approach: optimize the time-domain directly but use the
    spectral insight for initialization and gradient computation.
    """
    # Actually, let's stick with time-domain optimization but use a better initialization
    # and a more robust optimizer. The spectral approach is complex due to the
    # non-negativity constraint. Instead, let's use a multi-start strategy with
    # different initializations and L-BFGS-B.
    
    f = np.asarray(params, dtype=np.float64)
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10, np.zeros_like(f)
    
    conv = autoconvolve_fft(f)
    obj_raw = smooth_max(conv, beta)
    obj = obj_raw / (s * s)
    
    # Gradient computation (same as before)
    x_max = np.max(conv)
    if x_max < 1e-15:
        grad_conv = np.zeros_like(conv)
    else:
        x_scaled = conv / x_max
        shifted = x_scaled - np.max(x_scaled)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        grad_conv = weights
    
    n = 2 * N
    G = np.fft.rfft(grad_conv, n=n)
    F = np.fft.rfft(f, n=n)
    cross_corr = np.fft.irfft(np.conj(G) * F, n=n)
    d_obj_raw_df = cross_corr[:N]
    
    grad = d_obj_raw_df / (s * s) - 2.0 * obj_raw / (s ** 3)
    
    return obj, grad


def construct():
    start_time = time.time()
    
    # Multiple initializations
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Cosine bump
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    f_cos = np.cos(np.pi * x / 0.25) ** 2
    f_cos = np.maximum(f_cos, 0)
    candidates.append(f_cos)
    
    # 3. Gaussian
    f_gauss = np.exp(-10 * (x / 0.25) ** 2)
    candidates.append(f_gauss)
    
    # 4. Random smooth
    rng = np.random.RandomState(42)
    f_rand = rng.random(N)
    kernel = np.ones(5) / 5
    f_rand = np.convolve(f_rand, kernel, mode='same')
    candidates.append(f_rand)
    
    # 5. Two bumps
    f_two = np.zeros(N)
    centers = [-0.1, 0.1]
    for c in centers:
        mask = np.abs(x - c) <= 0.08
        f_two[mask] += np.cos(np.pi * (x[mask] - c) / 0.16) ** 2
    candidates.append(f_two)
    
    # 6. Flat top with rounded edges
    f_flat = np.zeros(N)
    f_flat[100:500] = 1.0
    # Smooth edges
    edge_kernel = np.array([0.25, 0.75, 1.0, 0.75, 0.25])
    f_flat = np.convolve(f_flat, edge_kernel, mode='same')
    candidates.append(f_flat)
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for f in candidates:
        s = np.sum(f)
        if s > 0:
            f_norm = f / s * N
            score = compute_score(f_norm)
            if score < best_score:
                best_score = score
                best_f = f_norm.copy()
    
    # Use L-BFGS-B with multiple beta values
    if time.time() - start_time < TIME_LIMIT - 30:
        beta_values = [5.0, 10.0, 20.0]
        for beta in beta_values:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            def obj_func(f):
                obj, _ = objective_freq(f, beta)
                return obj
            
            def grad_func(f):
                _, grad = objective_freq(f, beta)
                return grad
            
            # Constraints: f >= 0
            bounds = [(0, None)] * N
            
            try:
                result = minimize(
                    obj_func,
                    best_f,
                    jac=grad_func,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                if result.success or result.fun < best_score:
                    f_new = np.maximum(result.x, 0.0)
                    s = np.sum(f_new)
                    if s > 1e-15:
                        f_new = f_new / s * N
                        score = compute_score(f_new)
                        if score < best_score:
                            best_score = score
                            best_f = f_new.copy()
            except Exception:
                pass
    
    # Final normalization and cleanup
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s * N
    
    return [float(x) for x in best_f]


if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
