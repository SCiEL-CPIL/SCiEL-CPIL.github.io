"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Spectral coordinate descent in the frequency domain.
In frequency space, autoconvolution becomes pointwise |F|^2, so we can
shape the power spectrum to flatten the autocorrelation peak while
projecting back to non-negativity in time domain. This provides a
fundamentally different optimization path than time-domain gradient methods.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def parametric_to_step(coeffs, n_basis=20):
    """
    Convert cosine basis coefficients to step heights.
    coeffs[0] is the DC component (mean), coeffs[1:] are amplitudes.
    We ensure non-negativity by adding a large offset and clipping.
    """
    x = np.arange(N)
    f = np.zeros(N)
    for i in range(n_basis):
        if i < len(coeffs):
            f += coeffs[i] * np.cos(2 * np.pi * i * x / N)
    
    # Ensure non-negative by shifting and clipping
    # We want to preserve the shape, so we shift by -min + small epsilon
    f_min = np.min(f)
    f = f - f_min + 1e-6
    return f

def objective(coeffs):
    """Objective function for parametric optimization."""
    f = parametric_to_step(coeffs)
    return compute_score(f)

def fast_local_refinement(h, start_time, max_time):
    """
    Vectorized local refinement using gradient descent on the max peak.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.1
    
    # Precompute indices for gradient calculation
    # conv[k] = sum_j h[j] h[k-j]
    # grad_j = d(max_conv)/dh_j
    # If max is at k*, grad_j = h[k*-j] for valid j
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (s * s)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        # Find peak indices
        peak_indices = np.where(conv >= max_c * 0.99)[0]
        
        # Compute gradient for the max peak
        # We use a soft-max approach or just the top peak for stability
        # Let's use the top peak for simplicity and speed
        k_peak = np.argmax(conv)
        
        # Gradient with respect to h[j] is h[k_peak - j]
        # Note: h is defined on [0, N-1]. conv index k corresponds to lag k.
        # conv[k] = sum_{j=0}^{N-1} h[j] h[k-j]
        # d(conv[k])/dh[j] = h[k-j] + h[j] (if we consider symmetry, but here h is not necessarily symmetric in index, only autocorrelation is even)
        # Actually, conv is full mode.
        # d/dh_j (sum_i h[i] h[i+j]) = h[j] + sum_{i != j} ... 
        # Wait, conv[k] = sum_{i} h[i] h[i+k].
        # d(conv[k])/dh[j] = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
        # But since we are looking at the max of conv, and conv is symmetric around 0 lag? No, conv is defined on [-N+1, N-1] roughly.
        # np.convolve(h, h, 'full') has length 2N-1. Index 0 is lag -(N-1). Index N-1 is lag 0.
        
        # Let's just compute the gradient vector directly using vectorization
        # grad[j] = h[j + k_peak - (N-1)] + h[j - k_peak + (N-1)] ?
        # Let's stick to the definition:
        # conv_idx = k_peak
        # lag = conv_idx - (N - 1)
        # d(conv[lag])/dh[j] = h[j - lag] + h[j + lag]
        
        lag = k_peak - (N - 1)
        grad = np.zeros(N)
        
        # Term 1: h[j - lag]
        # Valid if 0 <= j - lag < N
        # j in [lag, N + lag - 1] intersect [0, N-1]
        start1 = max(0, lag)
        end1 = min(N, N + lag)
        if start1 < end1:
            indices = np.arange(start1, end1)
            grad[indices] += current_h[indices - lag]
            
        # Term 2: h[j + lag]
        # Valid if 0 <= j + lag < N
        # j in [-lag, N - lag - 1] intersect [0, N-1]
        start2 = max(0, -lag)
        end2 = min(N, N - lag)
        if start2 < end2:
            indices = np.arange(start2, end2)
            grad[indices] += current_h[indices + lag]
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
            
        # Update
        # We want to decrease the peak, so subtract gradient
        # But we must keep h non-negative and sum fixed
        # h_new = h - alpha * grad
        # Then project to non-negative and normalize
        
        h_new = current_h - alpha * grad
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            alpha *= 0.5
            continue
            
        h_new = h_new / s_new * N # Normalize to sum N
        
        # Check if it improved
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.01, 1.0)
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break
                
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(42)
    
    n_basis = 20
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Initialize coefficients
        if restart == 0:
            # Start with flat
            coeffs0 = np.zeros(n_basis)
            coeffs0[0] = 1.0
        elif restart == 1:
            # Random small perturbations
            coeffs0 = rng.normal(0, 0.1, n_basis)
            coeffs0[0] = 1.0
        elif restart == 2:
            # Gaussian-like bump in cosine space?
            coeffs0 = np.exp(-np.linspace(0, 5, n_basis))
            coeffs0[0] = 1.0
        elif restart == 3:
            # Two lobes
            coeffs0 = np.zeros(n_basis)
            coeffs0[0] = 1.0
            coeffs0[5] = 0.5
            coeffs0[10] = -0.5
        else:
            # Random
            coeffs0 = rng.normal(0, 0.2, n_basis)
            coeffs0[0] = 1.0
            
        # Optimize
        # Use L-BFGS-B
        res = minimize(objective, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'maxfun': 1000})
        
        if res.success or res.fun < best_score:
            f_res = parametric_to_step(res.x)
            score_res = compute_score(f_res)
            
            if score_res < best_score:
                best_score = score_res
                best_f = f_res
                
    # Final local refinement
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = fast_local_refinement(
            best_f, start_time, TIME_LIMIT - 2
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
