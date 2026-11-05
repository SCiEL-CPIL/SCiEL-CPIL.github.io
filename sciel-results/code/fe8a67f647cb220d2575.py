"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 400


def construct():
    # New approach: Minimize the peak of the autoconvolution relative to the sum.
    # We fix sum(h) = 1 to handle the denominator.
    # We minimize a smooth approximation of max(conv(h, h)).
    
    # Initialize with random values
    np.random.seed(42)
    h = np.random.randn(N)
    
    # Normalize to sum 1
    h = h / np.sum(h)
    
    # Define the objective function: sum(C^p) approximates max(C) for large p
    p = 50
    
    def objective(h_flat):
        h = h_flat.reshape(N)
        # Enforce sum(h) = 1 approximately? 
        # It's better to let the optimizer work on the constrained space or project.
        # Let's just compute the value. If sum is not 1, we normalize inside?
        # No, if we normalize inside, the gradient is messy.
        # Let's project in the gradient step or use a penalty.
        # Simpler: Use a parameterization that ensures sum=1.
        # h = 1/N + v, where sum(v)=0.
        # This is complex. Let's just use the projection method in a manual loop.
        
        C = np.convolve(h, h)
        return np.sum(C**p)
    
    # Manual gradient descent with projection
    h_curr = h.copy()
    lr = 0.01
    
    for _ in range(200):
        C = np.convolve(h_curr, h_curr)
        
        # Gradient of sum(C^p)
        # C_k = sum_i h_i h_{i+k}
        # d/dh_j (C_k^p) = p * C_k^(p-1) * (h_{j+k} + h_{j-k})
        
        grad = np.zeros(N)
        # We can compute this efficiently using FFT or just loops if N is small.
        # N=400 is small enough for O(N^2) or O(N^2 log N).
        # Let's use a vectorized approach if possible, or just a loop over lags.
        
        # Actually, let's use the property that C is the auto-correlation.
        # Gradient of sum(C_k^p) w.r.t h is related to the derivative of the correlation.
        
        # Simpler: Finite difference? No, too slow.
        # Analytic:
        # grad_j = sum_k p * C_k^(p-1) * (h_{j+k} + h_{j-k})
        
        # We can compute this by convolving the "weights" W_k = p * C_k^(p-1) with h.
        # Note: C_k is defined for k in [-(N-1), N-1].
        # np.convolve gives full convolution.
        # Let's construct the full C array.
        
        C_full = np.convolve(h_curr, h_curr)
        # C_full indices: 0 to 2N-2.
        # Lag k corresponds to index k + (N-1) in a centered view?
        # np.convolve(h, h)[k] = sum_i h_i h_{i+k-(N-1)}? 
        # Standard: conv[0] = h0*h0. conv[N-1] = sum h_i h_i = C_0.
        # So C_0 is at index N-1.
        # C_k (lag k) is at index N-1+k.
        
        # We need weights W for each lag.
        # Let's create an array W of length 2N-1.
        W = p * (C_full ** (p-1))
        
        # The gradient w.r.t h_j is the convolution of W with h?
        # Let's verify.
        # L = sum_k C_k^p.
        # C_k = <h, shift_k(h)>.
        # dL/dh = sum_k p C_k^(p-1) dC_k/dh.
        # dC_k/dh_j = h_{j+k} + h_{j-k}.
        # This is equivalent to convolving the sequence of weights (symmetric) with h.
        
        # Since C_full is symmetric (auto-correlation), W is symmetric.
        # The gradient is np.convolve(h, W, mode='same')?
        # Let's check indices.
        # If we convolve h and W (full), the result at index N-1+j should be the gradient for h_j.
        
        grad_full = np.convolve(h_curr, W, mode='full')
        grad = grad_full[N-1 : N-1+N]
        
        # Update h
        h_curr = h_curr - lr * grad
        
        # Project to sum(h) = 1
        h_curr = h_curr - (np.sum(h_curr) - 1.0) / N
        
    return h_curr
