"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Direct optimization on bin values using FFT-accelerated autoconvolution.
Uses subgradient descent to reduce the peak autoconvolution, followed by
coordinate descent for local polishing.

This approach avoids the limitations of parametric families and directly
optimizes the 600 degrees of freedom.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_convolve(f):
    """Compute autoconvolution using FFT for speed."""
    n = len(f)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.fft(f, fft_size)
    C = np.fft.ifft(F * F)
    return C.real[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_convolve(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Initialize with a Gaussian bump centered at 0
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    sigma = 0.1
    f = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalize sum to 1 for stable optimization (score is scale invariant)
    f = f / np.sum(f)
    
    best_f = f.copy()
    best_score = compute_score(f)
    
    # Subgradient Descent Phase
    # We minimize max(conv) subject to sum(f)=1, f>=0.
    # Since score = max(conv) when sum(f)=1, we minimize max(conv).
    
    alpha = 0.05
    patience = 50
    no_improve = 0
    
    while time.time() - start_time < TIME_LIMIT - 20:
        conv = fft_convolve(f)
        max_val = np.max(conv)
        
        # Find lags with near-max value to form subgradient
        threshold = max_val * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute subgradient of max_val w.r.t. f
        # g[i] = sum_{m in peak_indices} d(conv[m])/d(f[i])
        # d(conv[m])/d(f[i]) = f[m-i] + f[i-m] (if indices valid)
        g = np.zeros(N)
        for m in peak_indices:
            for i in range(N):
                # Term 1: j=i in sum_j f[j]f[m-j] -> f[i]f[m-i]
                if 0 <= m - i < N:
                    g[i] += f[m - i]
                # Term 2: j=m-i in sum_j f[j]f[m-j] -> f[m-i]f[i]
                if 0 <= i - m < N:
                    g[i] += f[i - m]
                # Note: if i == m-i (i.e., m=2i), the term is f[i]^2, derivative is 2f[i].
                # The above logic adds f[i] twice, so it correctly gives 2f[i].
        
        # Normalize gradient to have unit L2 norm for stable step size
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        g = g / g_norm
        
        # Update f
        f_new = f - alpha * g
        
        # Enforce non-negativity and normalization
        f_new = np.maximum(f_new, 0.0)
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            break
        f_new = f_new / s_new
        
        # Check improvement
        score_new = compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            no_improve = 0
            # Increase alpha slightly if improving
            alpha = min(alpha * 1.05, 0.2)
        else:
            no_improve += 1
            # Decrease alpha if not improving
            alpha *= 0.9
            if no_improve > patience:
                break
            # Try to recover if stuck
            f = f_new # keep the new state even if score didn't improve, as it might be a saddle
    
    # Coordinate Descent Polish Phase
    # For each bin i, optimize f[i] to minimize max(conv)
    
    if time.time() - start_time < TIME_LIMIT - 5:
        # Use best_f found so far
        f = best_f.copy()
        current_score = best_score
        
        # Number of passes
        max_passes = 10
        
        for _ in range(max_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                # Save original value
                f_orig = f[i]
                
                # Compute conv without f[i] contribution?
                # Easier: just evaluate score for different f[i]
                # Since N=600, we can do a small 1D search.
                
                # The function to minimize is max(conv)
                # conv[m] = 2 * f[i] * f[m-i] + rest[m] (approx)
                # We can find the optimal f[i] by ternary search on [0, max_possible]
                
                # Upper bound for f[i]:
                # If f[i] is too large, it will dominate the peak.
                # A reasonable upper bound is 2 * max(f)
                max_f = np.max(f)
                upper_bound = 2 * max_f + 1e-6
                
                # Ternary search
                a = 0.0
                b = upper_bound
                iterations = 20
                for _ in range(iterations):
                    m1 = a + (b - a) / 3
                    m2 = b - (b - a) / 3
                    
                    f[i] = m1
                    s = np.sum(f)
                    if s < 1e-15: continue
                    f_norm = f / s
                    score1 = compute_score(f_norm)
                    
                    f[i] = m2
                    s = np.sum(f)
                    if s < 1e-15: continue
                    f_norm = f / s
                    score2 = compute_score(f_norm)
                    
                    if score1 < score2:
                        b = m2
                    else:
                        a = m1
                
                # Set to midpoint
                f[i] = (a + b) / 2
                s = np.sum(f)
                if s < 1e-15: break
                f = f / s
                score_final = compute_score(f)
                
                if score_final < current_score:
                    current_score = score_final
                    best_f = f.copy()
                    improved = True
                else:
                    # Revert if it got worse (shouldn't happen with ternary search on convex, but safe)
                    f[i] = f_orig
                    s = np.sum(f)
                    if s > 1e-15:
                        f = f / s
            
            if not improved:
                break
    
    # Final normalization to ensure sum is not 0
    s_final = np.sum(best_f)
    if s_final > 1e-15:
        best_f = best_f / s_final
    
    # Return as list of floats
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
