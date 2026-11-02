"""
Minimize the normalized maximum of autoconvolution for a nonnegative step function
with 600 equal intervals on [-1/4, 1/4].

Strategy: Use disjoint support / sparse binary patterns to exploit destructive
interference in the autoconvolution. The flat function achieves Cauchy-Schwarz
bound, but discrete step functions with carefully chosen support can potentially
reduce the normalized peak. We search over structured sparse patterns using
greedy and random methods.
"""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def _autoconv(h):
    """Compute full autoconvolution of h."""
    return fftconvolve(h, h[::-1], mode='full')

def _normalized_max_autoconv(h):
    """Compute normalized max of autoconvolution: max(ac) / (sum(h))^2."""
    s = np.sum(h)
    if s <= 0:
        return 1e10
    ac = _autoconv(h)
    peak = np.max(ac)
    return peak / (s * s)

def construct():
    t_start = time.time()
    
    # Flat function baseline
    flat = np.ones(N)
    flat_score = _normalized_max_autoconv(flat)
    best_h = flat.copy()
    best_score = flat_score
    
    # Try sparse binary patterns: select k intervals to be 1, rest 0
    # The autoconvolution of a binary pattern with k ones has peak at most k
    # (at lag 0), and sum = k, so normalized max <= k/k^2 = 1/k
    # For k < N, 1/k > 1/N, so this is worse... unless the peak is less than k
    
    # Actually for binary with k ones, ac[0] = k, so normalized = k/k^2 = 1/k.
    # This is worse than flat (1/N) for k < N.
    
    # Try weighted patterns: some intervals have height 1, others 0
    # But with non-uniform weights, we might reduce the peak at non-zero lags
    
    # Key insight: if we use a pattern where the autoconvolution is more flat,
    # the maximum might be reduced relative to the lag-0 value.
    
    # Try random smooth patterns with optimization
    rng = np.random.default_rng(42)
    
    # Approach: Use a low-dimensional parameterization
    # h_i = max(0, sum_j a_j * phi_j(x_i))
    # where phi_j are basis functions
    
    x = np.linspace(-0.25, 0.25, N)
    
    # Try Fourier-like basis
    def fourier_pattern(coeffs):
        h = np.zeros(N)
        for k, c in enumerate(coeffs):
            if k == 0:
                h += c
            else:
                h += c * np.cos(2 * np.pi * k * x)
        return np.maximum(h, 0)
    
    # Random restarts with gradient-free optimization
    n_restarts = 50
    n_coeffs = 10
    
    for r in range(n_restarts):
        if time.time() - t_start > DEADLINE:
            break
        
        # Random initial coefficients
        coeffs = rng.normal(0, 1, n_coeffs)
        coeffs[0] = 1.0  # Ensure positive bias
        
        best_local = np.inf
        best_local_coeffs = coeffs.copy()
        
        # Simple hill climbing
        for step in range(50):
            if time.time() - t_start > DEADLINE:
                break
            
            h = fourier_pattern(best_local_coeffs)
            s = np.sum(h)
            if s <= 0:
                break
            h_norm = h / s * N  # Normalize to same total mass as flat
            score = _normalized_max_autoconv(h_norm)
            
            if score < best_local:
                best_local = score
                best_local_coeffs = best_local_coeffs.copy()
            
            # Random perturbation
            new_coeffs = best_local_coeffs + rng.normal(0, 0.1, n_coeffs)
            new_coeffs[0] = max(new_coeffs[0], 0.5)
            
            h_new = fourier_pattern(new_coeffs)
            s_new = np.sum(h_new)
            if s_new <= 0:
                continue
            h_new_norm = h_new / s_new * N
            score_new = _normalized_max_autoconv(h_new_norm)
            
            if score_new < best_local:
                best_local = score_new
                best_local_coeffs = new_coeffs.copy()
        
        if best_local < best_score:
            best_score = best_local
            h_final = fourier_pattern(best_local_coeffs)
            s_final = np.sum(h_final)
            if s_final > 0:
                best_h = h_final / s_final * N
    
    # Also try structured sparse patterns
    # Select a random subset of intervals
    for r in range(30):
        if time.time() - t_start > DEADLINE:
            break
        
        k = rng.integers(100, N)
        indices = rng.choice(N, size=k, replace=False)
        h_sparse = np.zeros(N)
        h_sparse[indices] = 1.0
        
        # Try to optimize weights on these indices
        weights = np.ones(k)
        for step in range(20):
            if time.time() - t_start > DEADLINE:
                break
            
            h_test = np.zeros(N)
            h_test[indices] = weights
            s = np.sum(h_test)
            if s <= 0:
                break
            h_norm = h_test / s * N
            score = _normalized_max_autoconv(h_norm)
            
            if score < best_score:
                best_score = score
                best_h = h_norm.copy()
            
            # Perturb weights
            weights = np.maximum(weights + rng.normal(0, 0.1, k), 0.01)
    
    # Final check
    best_h = np.maximum(best_h, 0.0)
    final_score = _normalized_max_autoconv(best_h)
    
    # If we didn't improve, return flat
    if final_score >= flat_score:
        return [float(x) for x in flat]
    
    return [float(x) for x in best_h]
