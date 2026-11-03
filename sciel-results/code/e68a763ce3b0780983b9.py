"""Rewritten program: model step heights as a bounded, mean-0.5 sequence generated from a smooth
low-frequency basis plus structured noise, then optimize a soft-max surrogate of the maximum overlap
with shifted 1-h using projected gradient descent with a deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 110.0  # safety margin below the 120s wall-clock kill

def _overlap_max(h):
    """Compute the maximum absolute overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize max_k overlap_k.
    """
    one_minus_h = 1.0 - h
    # Compute circular cross-correlation: overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is equivalent to np.convolve style but circular.
    # Use FFT for efficiency.
    H = np.fft.fft(h)
    OH = np.fft.fft(one_minus_h)
    # Circular cross-correlation: overlap[k] = sum_i h[i] * oh[(i+k) % N]
    # = ifft( H * conj(OH) )? Let's be careful.
    # Actually, sum_i h[i] * oh[(i+k)%N] is the circular correlation.
    # np.correlate in circular mode: ifft(fft(h) * conj(fft(oh))) gives sum_i h[i] * oh[(i-k)%N]
    # We want sum_i h[i] * oh[(i+k)%N] = sum_i h[i] * oh[(i-k)%(-k)] ... 
    # Let's just use a direct approach for clarity since N=100 is small, or use fft properly.
    # For N=100, direct is fine and less error-prone.
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return np.max(overlaps)

def _soft_max_overlap(h, beta):
    """Soft-max approximation of the maximum overlap."""
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    m = np.max(overlaps)
    return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (overlaps - m))))

def _soft_max_grad(h, beta):
    """Gradient of the soft-max overlap with respect to h."""
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    m = np.max(overlaps)
    weights = np.exp(beta * (overlaps - m)) / np.sum(np.exp(beta * (overlaps - m)))
    
    grad = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        # d/dh_i [ sum_j h[j] * oh[(j+k)%N] ] = oh[(i+k)%N] = (1-h)[(i+k)%N]
        # But we also need d/dh_i of oh which is -1, so there's a -h[(i+k)%N] term.
        # Wait, overlap_k = sum_j h[j] * (1 - h[(j+k)%N])
        # d/dh_i overlap_k = (1 - h[(i+k)%N]) + sum_j h[j] * (-delta_{i, (j+k)%N})
        # = (1 - h[(i+k)%N]) - h[(i-k)%N]
        term1 = 1.0 - h[(np.arange(N) + k) % N]
        term2 = -h[(np.arange(N) - k) % N]
        grad += weights[k] * (term1 + term2)
    return grad

def _project(h):
    """Project h onto the box [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum is exactly N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if abs(current_sum - target_sum) > 1e-12:
        # Simple uniform adjustment, then re-clip and iterate a few times
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            # Add diff/N to all, then clip
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    
    # Initial guess: smooth low-frequency cosine
    t = np.arange(N)
    h = 0.5 + 0.3 * np.cos(2 * np.pi * 1 * t / N)
    h = _project(h)
    
    best_h = h.copy()
    best_val = _overlap_max(best_h)
    
    beta = 50.0
    lr = 0.01
    
    # Adaptive beta progression
    beta_schedule = [50, 100, 200, 500, 1000, 2000]
    beta_idx = 0
    
    for step in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Update beta every 500 steps
        if step > 0 and step % 500 == 0 and beta_idx < len(beta_schedule) - 1:
            beta_idx += 1
            beta = beta_schedule[beta_idx]
            lr = 0.01 / (1 + beta / 100.0)
        
        # Compute gradient
        grad = _soft_max_grad(h, beta)
        
        # Gradient descent step
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        # Check if improved
        val = _soft_max_overlap(h_new, beta)
        val_exact = _overlap_max(h_new)
        
        if val_exact < best_val:
            best_h = h_new.copy()
            best_val = val_exact
        
        h = h_new
    
    # Final polish with high beta and tiny LR
    if time.time() - start_time < DEADLINE - 5:
        h = best_h.copy()
        beta = 2000.0
        lr = 0.001
        for step in range(1000):
            if time.time() - start_time > DEADLINE:
                break
            grad = _soft_max_grad(h, beta)
            h_new = h - lr * grad
            h_new = _project(h_new)
            val_exact = _overlap_max(h_new)
            if val_exact < best_val:
                best_h = h_new.copy()
                best_val = val_exact
            h = h_new
    
    return best_h
