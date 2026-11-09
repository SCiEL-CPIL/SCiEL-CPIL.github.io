import numpy as np
from scipy.fft import fft, ifft
import time

N = 600
DEADLINE = 110.0  # Safety margin before 120s kill

def _autoconv_peak(h):
    """
    Compute the maximum of the normalized autoconvolution of step heights h.
    
    The autoconvolution of a step function with equal intervals is proportional to
    the circular convolution (for periodic extension) or linear convolution.
    Since the function is on [-1/4, 1/4] with 600 intervals, and we're looking at
    autoconvolution, we use FFT-based linear convolution.
    
    The normalized maximum: max of (autoconv / total_mass) where total_mass = sum(h) * interval_width
    But since we're minimizing the normalized max, and the problem says "normalized maximum of its autoconvolution",
    we interpret this as: max_k (sum_i h_i * h_{i+k}) / (sum h_i)^2 * N  or similar.
    
    Actually, for a step function f(x) = h_i on interval i, the autoconvolution is:
    (f*f)(x) = integral f(t) f(x-t) dt
    
    For step functions with equal width w = 1/300, the discrete autoconvolution at lag k is:
    c_k = w * sum_i h_i * h_{i+k}  (for appropriate range of k)
    
    The "normalized maximum" likely means max(c_k) / (integral f)^2 or max(c_k) / (sum h_i * w)^2
    But since we want an upper bound on C1, and the constant is scale-invariant in a certain way,
    let's define: score = max_k (sum_{i} h_i h_{i+k}) / (sum h_i)^2 * N
    
    Actually, let me think more carefully. The autoconvolution of f with itself:
    (f*f)(x) = ∫ f(t) f(x-t) dt
    
    If f is a step function with heights h_0,...,h_{N-1} on intervals of width w=1/N_total where N_total relates to the domain.
    Domain is [-1/4, 1/4], so total width = 1/2. With 600 intervals, w = (1/2)/600 = 1/1200.
    
    The autoconvolution at lag k (in units of w) is approximately:
    c_k = w * sum_{i} h_i * h_{i+k}
    
    The maximum of c_k over all k. To normalize, we divide by (integral f)^2 = (w * sum h_i)^2.
    
    So normalized_max = max_k [w * sum_i h_i h_{i+k}] / (w * sum h_i)^2
                     = max_k [sum_i h_i h_{i+k}] / (w * (sum h_i)^2)
    
    Hmm, but this depends on w. Let me reconsider. Perhaps the normalization is just:
    max_k (sum_i h_i h_{i+k}) / (sum h_i)^2
    
    This is scale-invariant (if we scale h by c, both numerator and denominator scale by c^2).
    """
    h = np.asarray(h, dtype=np.float64)
    total = np.sum(h)
    if total == 0:
        return 0.0
    # Linear autoconvolution
    ac = np.convolve(h, h, mode='full')
    # ac[k] = sum_i h[i] * h[k-i] for valid ranges
    # The max of ac divided by (sum h)^2
    max_ac = np.max(ac)
    return max_ac / (total * total)

def construct():
    start_time = time.time()
    
    # Strategy: Use frequency domain parameterization.
    # Let h = ifft(fft(h)) but parameterize via spectral amplitudes.
    # Actually, let's use a direct optimization approach with gradient descent.
    
    # Initialize with a smooth bump function (Gaussian-like)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h_best = np.exp(-50 * x**2)
    h_best = h_best / np.sum(h_best) * np.sqrt(N)  # Normalize
    
    score_best = _autoconv_peak(h_best)
    
    # Try a few different initializations
    inits = [
        np.ones(N),
        np.exp(-50 * x**2),
        np.exp(-10 * x**2),
        np.cos(np.pi * x * 4) ** 2,
        np.maximum(0, 1 - 4 * np.abs(x)),
    ]
    
    for h0 in inits:
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.maximum(h0, 0)
        if np.sum(h0) == 0:
            continue
        h0 = h0 / np.sum(h0)
        s = _autoconv_peak(h0)
        if s < score_best:
            score_best = s
            h_best = h0.copy()
    
    # Gradient descent in time domain with non-negativity constraint
    h = h_best.copy()
    lr = 0.01
    momentum = np.zeros_like(h)
    v = 0.9
    
    for step in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Compute gradient of autoconv_peak w.r.t. h
        # autoconv_peak = max_k ac_k / (sum h)^2
        # where ac_k = sum_i h_i h_{i+k}
        
        total = np.sum(h)
        ac = np.convolve(h, h, mode='full')
        max_idx = np.argmax(ac)
        max_val = ac[max_idx]
        
        # Gradient of ac_k w.r.t. h_i:
        # ac_k = sum_j h_j h_{k-j}
        # d(ac_k)/d(h_i) = h_{k-i} + h_{k-i} = 2*h_{k-i} (if k-i in range)
        # More carefully: ac_k = sum_j h_j * h_{k-j}
        # d/d(h_i): terms where j=i: h_{k-i}, and where k-j=i => j=k-i: h_i * 1 => h_{k-i}
        # So d(ac_k)/d(h_i) = h_{k-i} + h_{k-i} = 2*h_{k-i} if both indices valid
        
        # Actually for linear convolution: ac[k] = sum_{j} h[j] * h[k-j]
        # d(ac[k])/d(h[i]) = h[k-i] (from j=i term) + h[k-i] (from k-j=i term, i.e., j=k-i)
        # But we need to be careful with bounds.
        
        # Simpler: use numerical gradient or analytic
        # Let's use the fact that for the max term:
        # f = ac[max_idx] / total^2
        # df/d(h_i) = (d(ac[max_idx])/d(h_i) * total^2 - ac[max_idx] * 2*total * 1) / total^4
        #           = (d(ac[max_idx])/d(h_i)) / total^2 - 2*ac[max_idx] / (total^3)
        
        # d(ac[k])/d(h[i]) where ac = convolve(h, h, 'full')
        # ac[m] = sum_{j} h[j] * h[m-j] for m from 0 to 2N-2
        # h is indexed 0..N-1
        # d(ac[m])/d(h[i]) = h[m-i] (if 0 <= m-i < N) + h[m-i] (if 0 <= m-i < N)
        # Wait, that's not right. Let me redo.
        
        # ac[m] = sum_{j=0}^{N-1} h[j] * h[m-j] where h[k]=0 if k not in [0,N-1]
        # d(ac[m])/d(h[i]) = sum_j [d(h[j])/d(h[i]) * h[m-j] + h[j] * d(h[m-j])/d(h[i])]
        #                   = h[m-i] + h[m-i]  (when m-i is in [0,N-1])
        #                   = 2 * h[m-i] if 0 <= m-i < N, else 0
        
        k = max_idx
        grad_ac = np.zeros(N)
        # For each i, d(ac[k])/d(h[i]) = 2*h[k-i] if 0 <= k-i < N
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                grad_ac[i] = 2 * h[j]
        
        grad_f = grad_ac / (total * total) - 2 * max_val / (total * total * total)
        
        # Project gradient onto non-negative orthant (simple: just use gradient, 
        # then project h to be non-negative)
        momentum = v * momentum + (1 - v) * grad_f
        h = h - lr * momentum
        h = np.maximum(h, 0)  # Enforce non-negativity
        
        # Renormalize to keep sum constant (optional, since objective is scale-invariant)
        # Actually, the objective is scale-invariant, so we don't need to normalize.
        # But to avoid numerical issues, let's normalize.
        s = _autoconv_peak(h)
        if s < score_best:
            score_best = s
            h_best = h.copy()
    
    # Final check
    final_score = _autoconv_peak(h_best)
    if final_score < score_best:
        score_best = final_score
    
    # Return as list of floats
    return [float(x) for x in h_best]

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_autoconv_peak(result)}")
