import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _gradient_descent_init(n_iter=200, lr=0.05):
    """
    Use continuous gradient descent on L2 norm of autocorrelation sidelobes
    to find a good continuous solution, then threshold to binary.
    """
    # Start with random continuous values in [-1, 1]
    x = np.random.uniform(-1, 1, size=N)
    
    for it in range(n_iter):
        # Compute autocorrelation
        A = _calc_acf(x)
        
        # We want to minimize sum of squares of sidelobes (exclude A[0])
        # Gradient of 0.5 * sum(A[k]^2) w.r.t. x
        # A = x * x (circular convolution), so dA/dx = 2 * circ_conv(x, x)
        # Actually, for autocorrelation: A[k] = sum_i x[i] * x[i+k]
        # d(A[k]^2)/dx[j] = 2*A[k] * dA[k]/dx[j]
        # dA[k]/dx[j] = x[j-k] + x[j+k] (for circular)
        
        # More efficient: use FFT
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        
        # Gradient of L2 norm of sidelobes
        # d/dx (sum_k A[k]^2) where A = ifft(fft(x)^2)
        # Using chain rule through FFT: grad = ifft(2*P * fft(grad_A))
        # where grad_A[k] = 2*A[k] for k != 0, 0 for k=0
        
        A_full = np.fft.irfft(P, n=n_fft)
        A_full[0] = 0  # Exclude main lobe
        
        # Gradient: d/dx sum(A_full[k]^2) = 2 * ifft(2 * A_full * fft(x))
        # Actually: A_full = ifft(|F|^2), so dA_full/dx = 2 * ifft(F * conj(F) * ...)
        # Let's use: grad_x = 4 * ifft(F * conj(F) * A_full_conj)
        # Wait, let me think more carefully.
        
        # A = ifft(|F|^2) where F = fft(x)
        # dA/dx: A is a function of x through F
        # dA_k/dx_j = 2 * ifft(F * conj(F) * e^{-2pi i k j / n})_j ... this is getting complex
        
        # Simpler: use the fact that for autocorrelation, 
        # grad of sum_k A[k]^2 w.r.t. x = 4 * circ_conv(x, A)
        # where A has A[0] = 0
        
        # Compute circular convolution of x with A (with A[0]=0)
        A_for_grad = np.zeros(N)
        A_for_grad[1:] = A[1:]  # Keep sidelobes, zero out main lobe
        
        # Circular convolution via FFT
        Fx = np.fft.rfft(x, n=n_fft)
        FA = np.fft.rfft(A_for_grad, n=n_fft)
        grad = np.fft.irfft(4 * Fx * FA, n=n_fft)[:N]
        
        # Update
        x -= lr * grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1, 1)
    
    # Threshold to binary
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def _local_search(s, current_val, max_passes=50, time_limit=10.0, deadline=None):
    """Local search by flipping bits."""
    start = time.time()
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() - start < time_limit:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            
            old_si = s[i]
            s[i] = -old_si
            
            A_new = _calc_acf(s)
            val_new = _max_abs_acf_from_array(A_new)
            
            if val_new < current_val:
                current_val = val_new
                improved = True
            else:
                s[i] = old_si
    
    return s, current_val

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Strategy 1: Gradient descent initialization (multiple times)
    for init_iter in range(10):
        if time.time() > deadline - 5:
            break
        
        s = _gradient_descent_init(n_iter=150, lr=0.05)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=30, time_limit=5.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Strategy 2: Random restarts with refinement
    while time.time() < deadline - 2:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=20, time_limit=3.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
