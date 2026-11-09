import numpy as np
import time

N = 600
TIME_LIMIT = 145.0 # Safety margin

def compute_soft_max(c, alpha):
    """Compute smooth max of c."""
    c_max = np.max(c)
    # Handle overflow
    exp_c = np.exp(alpha * (c - c_max))
    log_sum_exp = np.log(np.sum(exp_c))
    return c_max + log_sum_exp / alpha

def compute_grad_soft_max(h, alpha):
    """Compute gradient of soft_max(h*h) w.r.t h."""
    c = np.fft.irfft(np.fft.rfft(h, n=2*N)**2, n=2*N)[:2*N-1]
    # Weights for soft max
    c_max = np.max(c)
    exp_c = np.exp(alpha * (c - c_max))
    w = exp_c / np.sum(exp_c)
    
    # Gradient is 2 * corr(w, h)
    # corr(w, h)_j = sum_k w_k h_{k-j}
    # This can be computed via FFT
    # Let's use linear convolution/correlation
    # np.correlate(w, h, mode='full')?
    # w is length 2N-1. h is length N.
    # We need the part corresponding to h indices 0..N-1.
    
    # Efficient way:
    # grad_h = np.zeros(N)
    # For j in 0..N-1:
    #   grad_h[j] = 2 * (sum_{k} w[k] * h[k-j] + sum_{k} w[k] * h[j-k])
    # Note: h is 0 outside 0..N-1.
    
    # Using FFT:
    # Let W = fft(w, 2N), H = fft(h, 2N)
    # corr(w, h) is ifft(W * conj(H)) ?
    # Cross correlation x * y: ifft(fft(x) * conj(fft(y)))
    
    nfft = 2 * (2*N - 1) # Safe padding
    W = np.fft.rfft(w, n=nfft)
    H = np.fft.rfft(h, n=nfft)
    
    # corr_wh = ifft(W * conj(H))
    # This gives correlation.
    corr_wh = np.fft.irfft(W * np.conj(H), n=nfft)
    
    # The correlation is defined as sum w[k] h[k-j].
    # The output of ifft(W*conj(H)) at index j is sum_k w[k] h[k-j] (circular).
    # With zero padding, it is linear.
    # We need the values for j=0..N-1.
    
    grad_h = 2.0 * corr_wh[:N]
    
    # Wait, the second term sum_k w[k] h[j-k] is corr(h, w) which is corr(w, h) reversed?
    # sum_k w[k] h[j-k]. Let m = j-k. sum_m w[j-m] h[m].
    # This is cross correlation of w and h, but with w reversed?
    # Actually, corr(w, h)_j = sum_k w_k h_{k-j}.
    # corr(h, w)_j = sum_k h_k w_{k-j}.
    # Since w is symmetric? No, w is not necessarily symmetric.
    # But c is symmetric (autoconvolution). w is symmetric around N-1?
    # c_k = c_{2N-2-k}? No, c_k for k=0..2N-2.
    # c_k = sum h_i h_{k-i}.
    # c_{2N-2-k} = sum h_i h_{2N-2-k-i}.
    # If h is symmetric, c is symmetric.
    # But h is not necessarily symmetric.
    # However, the gradient of the scalar function F(h) should be correct.
    
    # Let's just compute the gradient directly for safety if N is small, but N=600 is small enough for O(N^2)?
    # 600^2 = 360,000. Very fast.
    # Let's use direct computation to avoid FFT pitfalls with correlation directions.
    
    grad_h_direct = np.zeros(N)
    for j in range(N):
        # Term 1: sum_k w[k] h[k-j]
        # k-j in [0, N-1] => k in [j, j+N-1]
        # w is length 2N-1.
        k_start = j
        k_end = j + N
        if k_end > 2*N - 1: k_end = 2*N - 1
        # h indices: k-j => 0 to N-1
        # w indices: k
        grad_h_direct[j] += np.dot(w[k_start:k_end], h[:k_end-k_start])
        
        # Term 2: sum_k w[k] h[j-k]
        # j-k in [0, N-1] => k in [j-(N-1), j]
        k_start2 = j - (N - 1)
        if k_start2 < 0: k_start2 = 0
        k_end2 = j + 1
        # h indices: j-k.
        # Let m = j-k. m goes from j-k_start2 to j-k_end2?
        # If k = k_start2, m = j - k_start2.
        # If k = k_end2 - 1, m = j - (k_end2 - 1) = 1? No.
        
        # This loop is getting messy. Let's use FFT properly.
        
    return grad_h_direct # Placeholder
