"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(h):
    """Compute periodic autocorrelation of h."""
    n = len(h)
    # Use FFT for efficient computation
    f = np.fft.fft(h, 2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= n  # Normalize
    return acf

def _objective(h):
    """Compute the autocorrelation upper bound (max |acf[k]| for k>=1)."""
    acf = _compute_autocorr(h)
    # Exclude lag 0, take max absolute value
    return np.max(np.abs(acf[1:]))

def _gradient_subgrad(h):
    """Compute subgradient of max |acf[k]| for k>=1."""
    acf = _compute_autocorr(h)
    abs_acf = np.abs(acf[1:])
    max_val = np.max(abs_acf)
    
    # Subgradient: average over all k that achieve the max
    mask = (abs_acf >= max_val - 1e-10)
    if np.sum(mask) == 0:
        mask = np.zeros(len(acf) - 1, dtype=bool)
        mask[0] = True
    
    # For each k in mask, compute gradient of acf[k] w.r.t. h
    # acf[k] = (1/N) * sum_i h[i] * h[(i+k) mod N]
    # d(acf[k])/d(h[j]) = (1/N) * (h[(j-k) mod N] + h[(j+k) mod N]) for k>0
    # For k=0: d(acf[0])/d(h[j]) = 2*h[j]/N
    
    n = len(h)
    grad = np.zeros(n)
    for k in range(1, n):
        if mask[k - 1]:
            # acf[k] = (1/n) * sum_{i=0}^{n-1} h[i] * h[(i+k) % n]
            # d/dh[j] = (1/n) * (h[(j-k) % n] + h[(j+k) % n])
            grad += (h[(np.arange(n) - k) % n] + h[(np.arange(n) + k) % n]) / n
    
    # Normalize by number of active constraints
    grad /= np.sum(mask)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with a pseudo-random ±1 sequence
    rng = np.random.RandomState(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    best_h = h.copy()
    best_obj = _objective(best_h)
    
    # Subgradient descent
    lr = 0.1
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        obj = _objective(h)
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
        
        grad = _gradient_subgrad(h)
        h = h - lr * grad
        
        # Project back to ±1 (or keep continuous for now)
        # For signed step functions, we can keep continuous values
        # But ±1 might be optimal. Let's try both.
        
        # Adaptive learning rate
        if iteration % 1000 == 0:
            lr = max(0.01, lr * 0.95)
    
    # Final: project to ±1 if that improves
    h_proj = np.sign(best_h)
    obj_proj = _objective(h_proj)
    if obj_proj < best_obj:
        best_h = h_proj
        best_obj = obj_proj
    
    return best_h
# EVOLVE-BLOCK-END
