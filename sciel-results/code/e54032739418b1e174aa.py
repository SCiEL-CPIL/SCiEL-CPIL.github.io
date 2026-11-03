import numpy as np
import time

N = 600

def construct():
    # Deadline guard: 150s limit, aim for 140s to be safe
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = float('inf')
    
    # We can run many iterations. FFT of size 600 is very fast.
    # Let's try to run as many as possible.
    
    # Initial guess: Hann window
    n = np.arange(N)
    hann = 0.5 * (1 + np.cos(2 * np.pi * n / (N - 1)))
    hann = hann / np.sum(hann)
    c_hann = np.fft.irfft(np.fft.rfft(hann) ** 2)
    score_hann = np.max(c_hann)
    best_x = hann
    best_score = score_hann

    # Randomized search for flat-spectrum sequences
    # x = |IFFT(exp(i * phi))|
    
    trials = 0
    while time.time() < deadline:
        # Generate random phases
        phi = 2 * np.pi * np.random.rand(N)
        X = np.exp(1j * phi)
        
        # Compute IFFT
        x_complex = np.fft.ifft(X)
        x = np.abs(x_complex)
        
        # Normalize to sum=1
        s = np.sum(x)
        if s == 0:
            continue
        x = x / s
        
        # Compute autoconvolution (circular, which is fine for large N relative to support, 
        # or use linear if needed. The task says "autoconvolution", usually linear.
        # However, for step functions on a grid, circular autocorrelation is often the metric 
        # for "periodic" extensions. But "step function on [-1/4, 1/4]" implies compact support.
        # Linear convolution length is 2N-1.
        # Let's use linear convolution to be precise.
        
        # Optimization: Use FFT for linear convolution
        # Zero-pad to next power of 2 or 2*N
        n_fft = 2 * N
        X_fft = np.fft.rfft(x, n=n_fft)
        c = np.fft.irfft(X_fft * np.conj(X_fft), n=n_fft)
        
        # The linear autocorrelation is in c[0:N] (and c[n_fft-N+1:n_fft] is the other side)
        # Max is usually at lag 0 or near it.
        score = np.max(c)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        trials += 1
        
        # Break if we've done enough
        if trials > 100000: 
             break

    return best_x.tolist()
