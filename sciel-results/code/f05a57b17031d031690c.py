"""Spectral amplitude shaping for the first autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 600
DEADLINE = 115.0  # safety margin before 120s kill

def _compute_score(heights):
    """Compute the normalized maximum of the circular autoconvolution.
    
    The autoconvolution of a step function on [-1/4, 1/4] with N equal intervals
    corresponds to the circular convolution of the height vector with itself,
    scaled appropriately. We compute the circular autocorrelation (which is
    equivalent up to scaling for the max value) and normalize.
    
    For a step function f(x) = h_i on interval i, the autoconvolution at lag k
    is sum_i h_i * h_{i+k} (circular). The "normalized maximum" is max_k (autoconv[k]) / (mean of autoconv)
    or similar normalization. Let's think carefully.
    
    Actually, the autoconvolution of a function on [-1/4, 1/4] with itself:
    (f * f)(t) = integral f(x) f(t-x) dx
    
    For step functions with N equal intervals of width w = (1/2)/N = 1/(2N):
    (f*f)(t) at the discrete level corresponds to the circular convolution of h with itself,
    multiplied by w.
    
    The "normalized maximum" likely means: max of (f*f) / (integral of f * integral of f) or
    max of (f*f) / (mean value)^2 * something.
    
    Let me reconsider. The task says "minimize the normalized maximum of its autoconvolution".
    A common normalization: the autoconvolution peak / (L1 norm)^2 or peak / (L2 norm)^2.
    
    Given the context of "upper bound on C1" for an autocorrelation inequality, and the seed
    returns all 1s with score 1.0, let's compute what the seed gives.
    
    For h = [1]*N: circular autocorrelation at lag 0 is sum(h_i^2) = N.
    At lag k (k != 0), it's sum(h_i * h_{i+k}) = N (since all are 1).
    So the autocorrelation is constant = N.
    
    If normalized by (sum h)^2 / N = N^2/N = N, then ratio = N/N = 1. That matches!
    
    So the score is: max_k (circular_autocorr[k]) / (sum(h)^2 / N)
    = N * max_k (circular_autocorr[k]) / (sum(h))^2
    
    Wait, let me re-derive. For h=[1]*N:
    circular_autocorr[k] = sum_i h[i] * h[(i+k)%N] = N for all k.
    sum(h) = N.
    (sum(h))^2 / N = N^2/N = N.
    max(circular_autocorr) / ((sum(h))^2 / N) = N / N = 1. ✓
    
    So score = N * max_k(circular_autocorr[k]) / (sum(h))^2
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 1e10
    h = np.maximum(h, 0.0)
    
    # Circular autocorrelation via FFT
    nfft = 1 << (2*N - 1).bit_length()  # next power of 2 >= 2N-1
    H = np.fft.rfft(h, n=nfft)
    autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
    
    total = np.sum(h)
    if total < 1e-12:
        return 1e10
    
    score = N * np.max(autocorr) / (total * total)
    return float(score)


def _spectral_init():
    """Initialize with a smooth spectral profile."""
    # Start with a raised-cosine-like spectrum for good initial conditions
    freqs = np.fft.rfftfreq(N)
    # Smooth window in frequency domain
    w = 0.5 * (1 + np.cos(2 * np.pi * freqs * 0.5))
    h = np.fft.irfft(w * N, n=N)
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-10:
        h = np.ones(N)
    return h


def construct():
    start_time = time.time()
    
    # Try multiple initializations and optimization strategies
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Strategy 1: Gradient descent on the spectral magnitudes with zero phase
    # We parameterize h via its DFT: H[k] = |H[k]| * exp(i*phase[k])
    # With zero phase, h is real and even (symmetric).
    # Actually, for a real signal, H[-k] = conj(H[k]). With zero phase, H[k] is real,
    # so h is symmetric: h[i] = h[N-i].
    
    # Let's use a direct optimization approach with scipy
    # Objective: minimize score(h) subject to h >= 0
    # Use log-sum-exp smoothing for the max
    
    def smooth_score(h, beta=50.0):
        """Smooth approximation of the score using log-sum-exp for the max."""
        h = np.maximum(h, 0.0)
        nfft = 1 << (2*N - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
        
        total = np.sum(h)
        if total < 1e-12:
            return 1e10
        base = N / (total * total)
        
        # Log-sum-exp for max(autocorr)
        a = autocorr * base
        max_a = np.max(a)
        if max_a > 0:
            lse = max_a + np.log(np.sum(np.exp(a - max_a)))
        else:
            lse = 0.0
        return lse
    
    def grad_smooth_score(h, beta=50.0):
        """Gradient of the smooth score via finite differences or analytical."""
        # Use analytical gradient through FFT
        h = np.maximum(h, 0.0)
        nfft = 1 << (2*N - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
        
        total = np.sum(h)
        if total < 1e-12:
            return np.zeros(N)
        base = N / (total * total)
        
        # Gradient of log-sum-exp(a) w.r.t. a is softmax(a)
        a = autocorr * base
        max_a = np.max(a)
        exp_a = np.exp(a - max_a)
        softmax_a = exp_a / np.sum(exp_a)
        
        # da/dh: a = base * autocorr
        # autocorr[k] = sum_i h[i] h[(i+k)%N]
        # d(autocorr[k])/d(h[j]) = h[(j-k)%N] + h[(j+k)%N] (for circular)
        # Actually: autocorr[k] = sum_i h[i]*h[(i+k)%N]
        # d/dh[j]: term where i=j gives h[(j+k)%N], term where (i+k)%N=j gives i=(j-k)%N, h[(j-k)%N]
        # So d(autocorr[k])/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]
        
        # grad_a[j] = base * sum_k softmax_a[k] * (h[(j+k)%N] + h[(j-k)%N])
        # This is base * (softmax_a * h) circular convolution evaluated at j, times 2 (symmetric)
        # Actually: sum_k s[k] h[(j+k)%N] is circular correlation of s and h
        # sum_k s[k] h[(j-k)%N] is circular convolution of s and h
        
        # Both are the same if s is symmetric (which it is since autocorr is symmetric)
        # So grad_a = 2 * base * circular_corr(softmax_a, h)
        
        nfft2 = 1 << (2*N - 1).bit_length()
        S = np.fft.rfft(softmax_a, n=nfft2)
        H2 = np.fft.rfft(h, n=nfft2)
        # circular_corr(s, h)[j] = sum_k s[k] h[(j+k)%N] = ifft(fft(s) * conj(fft(h)))
        corr = np.fft.irfft(S * np.conj(H2), n=nfft2)[:N]
        
        grad_autocorr = 2.0 * base * corr
        
        # Also need gradient of base w.r.t. h
        # base = N / total^2, d(base)/d(h[j]) = -2*N/(total^3)
        d_base = -2.0 * N / (total**3)
        # score = base * max(autocorr) approx base * lse
        # Actually score = N * max(autocorr) / total^2
        # d(score)/d(h[j]) = N * d(max(autocorr))/d(h[j]) / total^2 + N*max(autocorr)*d(1/total^2)/d(h[j])
        # = grad_autocorr[j] + N * np.max(autocorr) * (-2/total^3)
        
        # For the smooth version:
        # score_smooth = base * lse where lse = logsumexp(base*autocorr)... 
        # This is getting complex. Let's just use finite differences for reliability.
        
        return None  # Skip analytical, use finite differences
    
    # Use scipy minimize with numerical gradients
    # Parameterize with h directly, constraint h >= 0
    # Use L-BFGS-B which handles bounds
    
    def objective(x):
        return smooth_score(x)
    
    # Multiple random starts
    np.random.seed(42)
    n_starts = 3
    
    for s in range(n_starts):
        if time_left() < 10.0:
            break
        
        if s == 0:
            h0 = _spectral_init()
        elif s == 1:
            # Slightly perturbed uniform
            h0 = np.ones(N) * (1 + 0.1 * np.random.randn(N))
            h0 = np.maximum(h0, 0.0)
        else:
            # Concentrated shape
            center = N // 2
            sigma = N / 4.0
            idx = np.arange(N)
            h0 = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
            h0 = np.maximum(h0, 0.0)
        
        # Normalize to have reasonable scale
        h0 = h0 / np.max(h0) if np.max(h0) > 0 else h0
        
        result = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        h_opt = np.maximum(result.x, 0.0)
        sc = _compute_score(h_opt)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # Polish step with tighter tolerance
        if time_left() > 15.0 and sc < best_score + 1e-6:
            result2 = minimize(
                lambda x: smooth_score(x, beta=100.0),
                h_opt,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            h_polish = np.maximum(result2.x, 0.0)
            sc2 = _compute_score(h_polish)
            if sc2 < best_score:
                best_score = sc2
                best_h = h_polish.copy()
    
    # Final check: also try a simple symmetric optimization
    if time_left() > 5.0:
        # Try optimizing a symmetric profile: h[i] = h[N-1-i]
        # This reduces dimensionality
        def obj_sym(x):
            h = np.zeros(N)
            h[:N//2] = x
            h[N//2:] = x[::-1] if N % 2 == 0 else np.concatenate([x[::-1], [0]])
            # Ensure proper symmetry
            h = np.maximum(h, 0.0)
            return smooth_score(h)
        
        x0 = best_h[:N//2].copy()
        if N % 2 == 1:
            x0 = best_h[:N//2].copy()
        else:
            x0 = best_h[:N//2].copy()
        
        res_sym = minimize(
            obj_sym,
            x0,
            method='L-BFGS-B',
            bounds=[(0, None)] * len(x0),
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        x_opt = np.maximum(res_sym.x, 0.0)
        h_sym = np.zeros(N)
        h_sym[:len(x_opt)] = x_opt
        if N % 2 == 0:
            h_sym[N//2:] = x_opt[::-1]
        else:
            h_sym[N//2:] = x_opt[::-1]
            h_sym[N//2] = 0  # middle element
        
        sc_sym = _compute_score(h_sym)
        if sc_sym < best_score:
            best_score = sc_sym
            best_h = h_sym.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]


if __name__ == '__main__':
    result = construct()
    print(f"Score: {_compute_score(result)}")
