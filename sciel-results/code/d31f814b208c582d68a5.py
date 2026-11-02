import numpy as np
import time

N = 100
HALF = N // 2
DEADLINE = 110.0

def _compute_overlaps(h):
    """
    Compute overlap for all shifts k.
    Overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    """
    H = np.fft.fft(h)
    # Autocorrelation C_k = sum_i h[i] h[(i+k)%N]
    # Note: np.fft.ifft(np.fft.fft(h) * np.fft.fft(h).conj()) gives circular autocorrelation
    # The shift in FFT corresponds to negative index or positive depending on convention.
    # C_k = sum h[i] h[i+k]. 
    # Let's verify.
    # If h is constant 0.5, C_k = N * 0.25.
    # Overlap_k = N/2 - C_k.
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # The autocorr[0] is sum h[i]^2. autocorr[k] is sum h[i]h[i-k] usually.
    # We need sum h[i]h[i+k]. This is autocorr[-k] or autocorr[N-k].
    # Since autocorr is real and even for real h? No, circular autocorr is symmetric C_k = C_{N-k}.
    # So autocorr[k] = sum h[i] h[i-k] = sum h[j] h[j+k] (by substitution j=i-k).
    # So autocorr[k] is exactly what we need for shift k.
    return HALF - autocorr

def _max_overlap_value(h):
    overlaps = _compute_overlaps(h)
    return float(np.max(overlaps))

def _grad_max_overlap(h):
    """
    Approximate gradient of max overlap.
    O_k = sum_i h_i - sum_i h_i h_{i+k}
    dO_k/dh_j = 1 - h_{j-k} (term where i=j) - h_{j+k} (term where i=j-k)
    For the max shift k*, gradient is [1 - h_{j-k*} - h_{j+k*}] for all j.
    We add a small smoothing to avoid discontinuity if multiple shifts are max.
    """
    overlaps = _compute_overlaps(h)
    # Soft max to get weights for each shift
    T = 0.05
    weights = np.exp(overlaps / T)
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    for k in range(N):
        if weights[k] < 1e-6:
            continue
        # dO_k/dh_j = 1 - h[(j-k)%N] - h[(j+k)%N]
        # Vectorized:
        # h_j - h_{j-k} - h_{j+k}
        # We can use roll
        term1 = np.ones(N)
        term2 = np.roll(h, k) # h[j-k]
        term3 = np.roll(h, -k) # h[j+k]
        # Wait, d/dh_j (h_j h_{j+k}) = h_{j+k}
        # d/dh_j (h_{j-k} h_j) = h_{j-k}
        # So dO_k/dh_j = 1 - h_{j+k} - h_{j-k}
        grad += weights[k] * (term1 - term3 - term2)
    return grad

def construct():
    t0 = time.time()
    
    # The constant solution h=0.5 is the global optimum for minimizing the maximum overlap.
    # Overlap_k = sum h_i (1-h_{i+k}) = N/2 - sum h_i h_{i+k}.
    # For h=0.5, sum h_i h_{i+k} = N/4 for all k.
    # Overlap_k = N/2 - N/4 = N/4.
    # It is proven that max Overlap_k >= N/4.
    # Thus, returning the constant vector is optimal.
    
    h = np.full(N, 0.5)
    
    return h
