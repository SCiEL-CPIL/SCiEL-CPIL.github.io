import numpy as np
from scipy.optimize import minimize
import time

N = 600

def compute_autocorr(h):
    """Compute discrete autocorrelation of step function heights."""
    h = np.asarray(h)
    R = np.correlate(h, h, mode='full')
    return R

def normalized_max_autocorr(h):
    """Compute normalized maximum of autoconvolution (peak sidelobe / main lobe)."""
    h = np.asarray(h)
    R = np.correlate(h, h, mode='full')
    main = R[N-1]  # zero lag
    if main <= 0:
        return 1.0
    # Find max absolute sidelobe
    sidelobes = np.concatenate([R[:N-1], R[N:]])
    max_side = np.max(np.abs(sidelobes))
    return max_side / main

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with a good initial guess
    # Try a few different initializations
    best_h = np.ones(N)
    best_val = normalized_max_autocorr(best_h)
    
    # Try a few analytical initializations
    # Triangular
    tri = np.abs(1 - 2*np.arange(N)/N)
    best_h = tri
    best_val = normalized_max_autocorr(best_h)
    
    # Cosine window
    cos_w = 0.5 * (1 + np.cos(np.pi * np.arange(N) / (N-1)))
    cos_w[cos_w < 0] = 0
    if np.sum(cos_w) > 0:
        cos_w = cos_w / np.sum(cos_w) * np.sum(best_h)
        v = normalized_max_autocorr(cos_w)
        if v < best_val:
            best_h = cos_w
            best_val = v
    
    # Use scipy minimize to reduce peak sidelobe
    def objective(x):
        return normalized_max_autocorr(x)
    
    def objective_and_grad(x):
        val = normalized_max_autocorr(x)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            grad[i] = (objective(xp) - objective(xm)) / (2*eps)
        return val, grad
    
    # Try gradient descent with restarts
    h = best_h.copy()
    
    # Normalize to fix scale (autocorr is scale-invariant in ratio)
    # Use a simple projected gradient method
    
    lr = 0.01
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        if restart == 0:
            h = best_h.copy()
        else:
            # Add noise for restart
            h = best_h.copy()
            h += 0.1 * np.random.randn(N)
            h = np.maximum(h, 0)
            if np.sum(h) == 0:
                h = best_h.copy()
        
        for it in range(2000):
            if time.time() > deadline - 20:
                break
            
            val = normalized_max_autocorr(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Compute gradient numerically (batch for speed)
            eps = 1e-5
            grad = np.zeros(N)
            # Use a subset of coordinates for speed if needed
            for i in range(0, N, 2):  # every other coordinate
                hp = h.copy()
                hp[i] += eps
                hm = h.copy()
                hm[i] -= eps
                grad[i] = (normalized_max_autocorr(hp) - normalized_max_autocorr(hm)) / (2*eps)
            
            # Gradient step with projection to nonnegative
            h = h - lr * grad
            h = np.maximum(h, 0)
            
            # Adaptive learning rate
            if it % 100 == 0:
                lr = max(0.001, lr * 0.9)
    
    # Final polish with smaller steps
    if time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.001
        for it in range(1000):
            if time.time() > deadline - 10:
                break
            val = normalized_max_autocorr(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            eps = 1e-6
            for i in range(N):
                hp = h.copy()
                hp[i] += eps
                hm = h.copy()
                hm[i] -= eps
                grad_i = (normalized_max_autocorr(hp) - normalized_max_autocorr(hm)) / (2*eps)
                h[i] = max(0, h[i] - lr * grad_i)
    
    return list(best_h)
