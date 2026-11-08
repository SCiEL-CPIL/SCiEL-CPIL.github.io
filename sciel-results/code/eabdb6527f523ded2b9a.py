import numpy as np
from scipy import signal

N = 600
dx = 1.0 / (2 * N) # interval length is 1/2, so dx = (1/2)/600 = 1/1200
x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2 # centers of intervals

def compute_value(h):
    # h is array of heights
    # Normalize
    s = np.sum(h)
    if s == 0: return float('inf')
    h_norm = h / s
    # Convolution
    # Using FFT for speed
    # Zero pad to 2*N
    f = np.fft.fft(h_norm, n=2*N)
    conv = np.fft.ifft(f * f).real
    # The convolution of two functions on [-1/4, 1/4] is on [-1/2, 1/2]
    # The discrete convolution result needs to be aligned correctly.
    # np.convolve(h, h, mode='full') gives length 2N-1.
    # FFT method:
    # The peak is usually at the center.
    # Let's just take the max of the real part.
    # Note: The continuous peak is approx by max(conv) * dx.
    # The value V = max(conv) * dx / (sum(h_norm)*dx)^2 = max(conv) * dx / (1*dx)^2 = max(conv)/dx.
    # Wait, sum(h_norm) = 1. So integral is dx.
    # V = (max_conv * dx) / (dx)^2 = max_conv / dx.
    # So we just need to minimize max_conv.
    return np.max(conv) / dx

def construct():
    # Scan Gaussian parameters
    best_h = None
    best_val = float('inf')
    
    # B controls width: exp(-B*x^2)
    # Small B -> wide (uniform-like)
    # Large B -> narrow (delta-like)
    for B in np.linspace(1, 50, 50):
        h = np.exp(-B * x**2)
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Local refinement using subgradient descent
    # We want to minimize max(conv)
    # Gradient of max(conv) at index k is grad_k
    # conv[m] = sum_i h[i] h[i+m]
    # d(conv[k])/d(h[j]) = h[j+k] + h[j-k]
    
    h = best_h
    h = h / np.sum(h) # Normalize
    
    for step in range(200):
        f = np.fft.fft(h, n=2*N)
        conv = np.fft.ifft(f * f).real
        k_max = np.argmax(conv)
        
        # Calculate gradient
        # We need h[j+k] and h[j-k]
        # Using np.roll
        # Note: indices in conv correspond to lags.
        # The peak is at lag k_max.
        # Due to symmetry, we can assume k_max is positive or handle both.
        # Actually, conv is symmetric.
        
        g1 = np.roll(h, -k_max)
        g2 = np.roll(h, k_max)
        grad = g1 + g2
        
        # Update
        eta = 0.01
        h_new = h - eta * grad
        h_new = np.clip(h_new, 0, None)
        
        # Renormalize
        s = np.sum(h_new)
        if s == 0:
            break
        h_new = h_new / s
        
        # Check if improved
        val_new = compute_value(h_new)
        if val_new < compute_value(h):
            h = h_new
        else:
            # Reduce eta
            eta *= 0.5
            if eta < 1e-6:
                break
                
    return h.tolist()
