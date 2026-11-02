import numpy as np
from scipy.fft import fft, ifft

N = 600

def construct():
    # Initialize with uniform distribution
    h = np.ones(N)
    h = h / np.sqrt(np.sum(h ** 2))
    
    best_h = h.copy()
    best_obj = np.max(np.convolve(h, h, mode='full'))
    
    # Hyperparameters
    lr = 0.05
    max_iters = 1000
    patience = 100
    no_improve = 0
    
    # Precompute indices for convolution peak check
    # We want to minimize the max of the convolution.
    # We use the L2 norm of the convolution as a smooth proxy for optimization,
    # but evaluate the actual peak for scoring.
    
    for it in range(max_iters):
        # 1. Compute FFT
        H = fft(h)
        
        # 2. Gradient of sum(|H|^4)
        # d/dH sum(|H|^4) = 4 |H|^2 H
        # We want to minimize this, so we subtract gradient
        grad_H = 4 * np.abs(H)**2 * H
        
        # 3. Update in Fourier domain
        H_new = H - lr * grad_H
        
        # 4. Inverse FFT
        h_new = ifft(H_new).real
        
        # 5. Project to non-negative
        h_new = np.maximum(h_new, 0)
        
        # 6. Normalize L2
        norm = np.sqrt(np.sum(h_new ** 2))
        if norm < 1e-12:
            h_new = h
        else:
            h_new = h_new / norm
        
        # Evaluate
        ac = np.convolve(h_new, h_new, mode='full')
        current_obj = np.max(ac)
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_h = h_new.copy()
            no_improve = 0
            # Maybe increase learning rate slightly if improving
            lr = min(lr * 1.01, 0.2)
        else:
            no_improve += 1
            if no_improve > patience:
                break
            # Decay learning rate
            lr *= 0.95
            
        h = h_new
        
    return list(best_h)
