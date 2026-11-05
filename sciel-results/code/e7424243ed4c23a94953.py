import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # Generate a random orthogonal matrix Q to define the "wavelet-like" basis
    # Using QR decomposition of a random Gaussian matrix
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(A)
    
    # Initialize wavelet coefficients c
    # Start with small random values to avoid sign ambiguity initially
    c = rng.standard_normal(N) * 0.1
    
    best_s = np.ones(N)
    best_val = np.inf
    
    # Helper to compute autocorrelation max
    def compute_autoconv_max(s):
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        return np.max(np.abs(r)) * w, r

    # Subgradient descent on c
    # Learning rate schedule
    eta = 1.0
    
    for iteration in range(2000):
        # Reconstruct s
        s_cont = Q @ c
        s = np.sign(s_cont)
        s[s == 0] = 1
        
        # Compute objective
        val, r = compute_autoconv_max(s)
        
        if val < best_val:
            best_val = val
            best_s = s.copy()
            
        # Find subgradient
        k_max = np.argmax(np.abs(r))
        r_max = r[k_max]
        
        if r_max == 0:
            break
            
        sign_r = np.sign(r_max)
        
        # Compute gradient of |r[k_max]| w.r.t s
        # r[k] = sum_{i=0}^{N-1-k} s[i] s[i+k]
        # grad[i] = sign(r) * (s[i+k] if i < N-k else 0) + sign(r) * (s[i-k] if i >= k else 0)
        
        grad_s = np.zeros(N)
        
        # Vectorized gradient calculation
        # Term 1: s[i+k] for i < N-k
        # This is a shift of s
        if k_max > 0:
            # grad_s[0:N-k_max] += s[k_max:N]
            grad_s[:N-k_max] += s[k_max:]
            # grad_s[k_max:N] += s[0:N-k_max]
            grad_s[k_max:] += s[:N-k_max]
        else:
            # k_max == 0, r[0] = sum(s^2) = N, grad is 2*s
            grad_s[:] = 2.0 * s
            
        grad_s *= sign_r
        
        # Map gradient to c space
        grad_c = Q.T @ grad_s
        
        # Update c
        # Decay learning rate
        eta = 0.5 / (1 + iteration * 0.01)
        c -= eta * grad_c
        
        # Optional: Add noise to escape local minima (Simulated Annealing style)
        if iteration % 100 == 0 and iteration > 0:
             c += rng.standard_normal(N) * 0.1 * (1 - iteration/2000.0)
             
    return best_s
