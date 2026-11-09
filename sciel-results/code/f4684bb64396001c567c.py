"""Rewritten program for the suite task; optimizing for correct metric."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Correct objective function based on inspirations: max(h*h) / ||h||_2^2
    def objective(h):
        h = np.maximum(h, 0)
        q = np.sum(h * h)
        if q < 1e-12:
            return 1e10
        # Linear autoconvolution
        # Using FFT for speed. 
        # Zero-pad to at least 2N-1. 2*N is a power of 2 if N is power of 2? 
        # N=600. 2*600 = 1200. Next power of 2 is 2048.
        n_fft = 2048
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)
        # The linear convolution is in conv[0:2N-1]
        # Actually, irfft returns length n_fft.
        # The linear convolution of two length N sequences is length 2N-1.
        # The first 2N-1 elements of the circular convolution (if padded enough) 
        # correspond to the linear convolution.
        # Since n_fft >= 2N-1, conv[0:2N-1] is the linear conv.
        peak = np.max(conv[:2*N-1])
        return peak / q

    def objective_and_grad(h):
        h = np.maximum(h, 0)
        q = np.sum(h * h)
        if q < 1e-12:
            return 1e10, np.zeros(N)
        
        n_fft = 2048
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)
        conv_lin = conv[:2*N-1]
        peak = np.max(conv_lin)
        val = peak / q
        
        # Find index of peak in linear convolution
        k_star = int(np.argmax(conv_lin))
        
        # Gradient of peak w.r.t h
        # conv_lin[k] = sum_i h[i] * h[i+k] (for i such that indices valid)
        # d(conv_lin[k])/dh[j] = h[j+k] + h[j-k] (if indices in [0, N-1])
        grad_peak = np.zeros(N)
        for j in range(N):
            val1 = 0.0
            val2 = 0.0
            idx1 = j + k_star
            idx2 = j - k_star
            if 0 <= idx1 < N:
                val1 = h[idx1]
            if 0 <= idx2 < N:
                val2 = h[idx2]
            grad_peak[j] = val1 + val2
            
        # Gradient of q = sum h^2 is 2h
        # d(val)/dh = (grad_peak * q - peak * 2h) / q^2
        grad = (grad_peak * q - peak * 2 * h) / (q * q)
        
        return val, grad

    best_h = np.ones(N)
    best_val = objective(best_h)
    
    # Try multiple random initializations
    n_starts = 15
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            h0 = np.ones(N)
        elif s == 1:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-x**2 * 4)
        elif s == 2:
            # Two peaks
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-(x-0.3)**2 * 10) + np.exp(-(x+0.3)**2 * 10)
        elif s == 3:
            # Cosine bump
            x = np.linspace(-1, 1, N)
            h0 = np.maximum(0, np.cos(np.pi * x / 2))
        elif s == 4:
            # Uniform with slight noise
            h0 = 1.0 + 0.1 * np.random.randn(N)
            h0 = np.maximum(h0, 0)
        else:
            h0 = np.random.rand(N)
            
        # Normalize to avoid scaling issues during optimization
        # The objective is scale invariant: V(alpha h) = V(h).
        # So we can normalize ||h||_2 = 1 or sum h = 1.
        # Let's normalize ||h||_2 = 1 to keep numbers stable.
        norm = np.linalg.norm(h0)
        if norm > 0:
            h0 = h0 / norm
        
        try:
            res = minimize(objective_and_grad, h0, method='L-BFGS-B', 
                          jac=True,
                          bounds=[(0, None)] * N,
                          options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 60:
            break
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
