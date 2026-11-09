import numpy as np

N = 600

def autoconvolve(h):
    """Compute autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H.conj(), n=size)
    return C[:2*n-1]

def construct():
    # Initialize H (rfft of z)
    n_fft = N // 2 + 1
    H = np.zeros(n_fft, dtype=np.complex128)
    # Start with flat amplitude, random phase (white noise)
    # This tends to produce a "random" h, which might be spiky.
    # We want to minimize max(c)/sum(h)^2, which favors uniform h.
    # Uniform h has H[0] = N, H[k>0] = 0.
    # So we should initialize H close to a delta in frequency (DC only)?
    # If H is DC only, z is constant, h is constant.
    # Let's start with H[0] = 1, others 0?
    # Then z = 1/N * 1? No, irfft of [1, 0...] is 1/N.
    # h = softplus(1/N) ~ 1/N.
    # This is a good start for uniformity.
    
    H[0] = 1.0
    # Add small noise to break symmetry and allow optimization
    H += (np.random.randn(n_fft) + 1j * np.random.randn(n_fft)) * 0.01
    # Ensure H[0] is real? rfft[0] is real.
    H[0] = np.real(H[0])
    
    # Adam optimizer
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(H)
    v = np.zeros_like(H)
    t = 0
    
    best_h = None
    best_score = np.inf
    
    for it in range(3000):
        z = np.fft.irfft(H, n=N)
        h = np.logaddexp(0, z)
        
        c = autoconvolve(h)
        
        Q = np.sum(h)
        if Q < 1e-10:
            continue
        M = np.max(c)
        score = M / (Q * Q)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Gradient
        k_star = np.argmax(c)
        
        # dM/dh
        grad_M = np.zeros(N)
        if k_star < N:
            # h_{j-k}
            grad_M[k_star:] += h[:N-k_star]
            # h_{j+k}
            grad_M[:N-k_star] += h[k_star:]
        else:
            # k_star >= N
            # c_k = sum_{i=0}^{N-1-k} h_i h_{i+k} ? No.
            # c_k = sum_i h_i h_{i-k}.
            # If k > 0, i must be >= k.
            # If k >= N, no overlap?
            # Linear convolution length 2N-1. Max index 2N-2.
            # If k_star >= N, the overlap is from the "other side"?
            # No, linear convolution indices are 0 to 2N-2.
            # c_k for k in [0, N-1] is sum_{i=k}^{N-1} h_i h_{i-k}.
            # c_k for k in [N, 2N-2] is sum_{i=0}^{2N-2-k} h_i h_{i-k}?
            # Let's use the property c_{2N-2-k} = c_k for symmetric? No, h is not symmetric.
            # But c_k = c_{-k} for real? No, autoconv is symmetric: c_k = c_{-k} in infinite domain.
            # In linear convolution, c_k = sum h_i h_{i-k}.
            # If k >= N, let k' = 2N-2-k.
            # Actually, just compute it directly if needed, but peak is usually near 0.
            # For non-negative h, peak is at k=0.
            # So k_star should be 0 (or N-1 in the array returned by autoconvolve? 
            # autoconvolve returns C[:2N-1].
            # C[0] is lag -N+1? No.
            # np.fft.irfft of |H|^2 gives the circular convolution of the padded signal.
            # The padded signal is h padded with N zeros.
            # The circular convolution of h_pad with itself has the linear convolution embedded.
            # The peak of linear autoconv is at index N-1 in the full 2N-1 array?
            # Let's check: h=[1,1]. pad=[1,1,0,0].
            # conv = [1, 2, 1, 0].
            # Peak is at index 1. N-1 = 1.
            # So k_star in the returned array `c` corresponds to lag `k_star - (N-1)`?
            # No, `c` is `C[:2N-1]`.
            # `C` is the inverse FFT.
            # `C[0]` is the sum of products with lag 0?
            # For `h=[1,1]`, `H=[2, 1-1j]`. `|H|^2 = [4, 2]`.
            # `irfft([4, 2])` -> `[1, 2, 1, 0]`?
            # Let's check `np.fft.irfft([4, 2], n=4)`.
            # `4 + 2 cos(0) + 2 cos(pi) ...`?
            # Actually, `C[0]` is the DC term? No.
            # `C[k]` is the coefficient.
            # For `h=[1,1]`, `c=[1,2,1,0]`.
            # `max(c)` is 2.
            # `k_star = 1`.
            # Lag is 0.
            # So `k_star` in the array is the index of the peak.
            # The lag is `k_star - (N-1)`?
            # If `k_star=1` and `N=2`, lag = 0.
            # If `k_star=0`, lag = -1.
            
            # So we need to map `k_star` to the actual lag `k`.
            # `k = k_star - (N - 1)`.
            
            k_lag = k_star - (N - 1)
            
            # Now compute grad_M for lag k_lag.
            # c_k = sum_i h_i h_{i-k}
            # If k_lag >= 0:
            #   i from k_lag to N-1.
            #   grad[j] += h[j-k] if j>=k
            #   grad[j] += h[j+k] if j+k < N
            # If k_lag < 0:
            #   let k_pos = -k_lag.
            #   i from 0 to N-1-k_pos.
            #   grad[j] += h[j+k_pos] if j+k_pos < N
            #   grad[j] += h[j-k_pos] if j>=k_pos
            
            if k_lag >= 0:
                k = k_lag
                if k < N:
                    grad_M[k:] += h[:N-k]
                    grad_M[:N-k] += h[k:]
            else:
                k = -k_lag
                if k < N:
                    grad_M[:N-k] += h[k:]
                    grad_M[k:] += h[:N-k]
        
        dQ_dh = np.ones(N)
        
        grad_score_h = (Q * grad_M - M * 2 * dQ_dh) / (Q * Q)
        
        grad_z = grad_score_h * (1 / (1 + np.exp(-np.clip(z, -50, 50))))
        
        grad_H = np.fft.rfft(grad_z, n=N)
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad_H
        v = beta2 * v + (1 - beta2) * (grad_H * grad_H)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        H -= lr * m_hat / (np.sqrt(v_hat) + eps)
        H[0] = np.real(H[0])
        
    if best_h is None:
        return [1.0] * N
    return list(best_h)
