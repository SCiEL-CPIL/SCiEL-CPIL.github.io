"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _max_overlap_fft(h):
    """
    Compute the maximum circular overlap of h with 1-h using FFT.
    overlap(s) = sum(h[i] * (1-h)[(i+s)%N])
    This is the cross-correlation of h and (1-h).
    """
    one_minus_h = 1.0 - h
    # Circular cross-correlation
    # corr[s] = sum_i h[i] * one_minus_h[(i+s) % N]
    # This is equivalent to convolving h with reversed one_minus_h?
    # np.correlate(h, one_minus_h, mode='full') gives linear correlation.
    # For circular, we can use FFT.
    # F = fft(h), G = fft(one_minus_h)
    # corr = ifft(F * conj(G))
    # Let's verify.
    # sum_i h[i] g[(i+s)%N] = sum_i h[i] sum_k g[k] delta_{(i+s)%N, k}
    # This is the standard circular cross-correlation.
    
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    # Cross-correlation is ifft(H * conj(G))
    corr = np.fft.ifft(H * np.conj(G)).real
    return np.max(corr)

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _objective_2d(params):
    """
    Objective function for 2D optimization of log(w) and c.
    We add a penalty for the sum not being N/2.
    """
    log_w, c = params
    w = np.exp(log_w)
    
    # Clamp w
    w = np.clip(w, 1e-3, N)
    
    h = _construct_h(w, c)
    
    # Penalty for sum constraint
    target_sum = N / 2.0
    current_sum = np.sum(h)
    penalty = (current_sum - target_sum) ** 2 * 100.0
    
    # Calculate overlap
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return 1000.0 + penalty
        
    ov = _max_overlap_fft(h)
    
    return ov + penalty

def construct():
    """
    Construct the sequence of step heights.
    """
    # Define bounds for differential evolution
    # log(w) from log(1e-3) to log(N)
    # c from -N to 2*N (to cover all shifts)
    bounds = [
        (np.log(1e-3), np.log(N)),
        (-N, 2 * N)
    ]
    
    # Use differential evolution for global search
    # Tolerance can be tight
    res = differential_evolution(
        _objective_2d, 
        bounds, 
        seed=42, 
        maxiter=200, 
        tol=1e-6,
        polish=True
    )
    
    best_log_w, best_c = res.x
    best_w = np.exp(best_log_w)
    
    h = _construct_h(best_w, best_c)
    
    # Final correction for exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    # Distribute difference
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)
                    
    return h
