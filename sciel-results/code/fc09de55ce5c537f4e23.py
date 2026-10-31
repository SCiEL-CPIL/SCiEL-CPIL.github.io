import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 600
DEADLINE = 140

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    # Precompute index array
    i = np.arange(N, dtype=float)
    
    # Stage 1: Differential Evolution on Cosine Basis
    # We represent h(x) = sum_{k=0}^{M} c_k cos(2*pi*k*x/N)
    # This ensures smoothness.
    M = 15
    # Bounds for coefficients
    # c0 is the mean, should be positive.
    # Other coeffs can be negative.
    bounds = [(0.1, 5.0)] + [(-2.0, 2.0)] * M
    
    def coeffs_to_h(c):
        h = np.zeros(N)
        for k in range(len(c)):
            h += c[k] * np.cos(2 * np.pi * k * i / N)
        return h

    def objective(coeffs):
        h = coeffs_to_h(coeffs)
        if np.any(h < 0):
            return 1e6
        s = np.sum(h)
        if s < 1e-10:
            return 1e6
        h_norm = h / s
        return compute_autoconv_max(h_norm)

    # Run Differential Evolution
    # Time budget: ~60 seconds
    # Check time before starting
    if time_left() > 70:
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=200, 
                popsize=15, 
                seed=42,
                tol=1e-5,
                init='sobol'
            )
            best_coeffs = result.x
        except Exception:
            best_coeffs = None
    else:
        best_coeffs = None

    # Initialize best_h
    if best_coeffs is not None:
        best_h = coeffs_to_h(best_coeffs)
        if np.any(best_h < 0):
             # Fallback if DE failed to find valid solution
             best_h = np.ones(N)
        else:
             best_h = best_h / np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    current_val = compute_autoconv_max(best_h)
    best_val = current_val

    # Stage 2: Projected Gradient Descent on full h vector
    # This allows breaking the smooth cosine constraint if it helps.
    if time_left() > 20:
        h = best_h.copy()
        eta = 0.01
        patience = 0
        max_patience = 50
        
        # Precompute FFT size
        size = 2 * N
        
        while time_left() > 5 and patience < max_patience:
            # Compute gradient
            conv = compute_autoconv(h)
            max_val = np.max(conv)
            
            # Subgradient of max: pick the index of the max
            # For better convergence, use a soft-max approximation
            beta = 50.0
            exp_conv = np.exp((conv - max_val) * beta)
            weights = exp_conv / np.sum(exp_conv)
            
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            
            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            # Update
            h_new = h - eta * grad_h
            
            # Project to feasible set: h >= 0, sum(h) = 1
            h_new = np.maximum(h_new, 0)
            s = np.sum(h_new)
            if s < 1e-10:
                break
            h_new = h_new / s
            
            new_val = compute_autoconv_max(h_new)
            
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
                h = h_new
                patience = 0
                eta = min(eta * 1.05, 0.1)
            else:
                patience += 1
                eta *= 0.9
                if eta < 1e-6:
                    break

    return best_h.tolist()
