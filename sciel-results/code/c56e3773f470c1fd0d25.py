import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 130  # safety margin before 150s limit

def _compute_score(h):
    """Compute the normalized maximum of the autoconvolution.
    For uniform h=1, sum=N, max_conv=N, score = N*N/N^2 = 1.
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    c = np.convolve(h, h, mode='full')
    return np.max(c) * N / (s * s)

def _smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def _local_refinement(h, start_time, max_time):
    """
    Local refinement using vectorized gradient-based peak reduction.
    Minimizes the peak of the autoconvolution by iteratively reducing contributions to peaks.
    """
    best_h = h.copy()
    best_val = _compute_score(h)
    current_h = h.copy()
    current_val = best_val
    
    # Ensure non-negative and normalized sum
    current_h = np.maximum(current_h, 0.0)
    s = np.sum(current_h)
    if s < 1e-15:
        current_h = np.ones(N)
        s = N
    current_h = current_h * (N / s)
    
    alpha = 0.5
    patience = 0
    max_patience = 50
    peak_fraction = 0.1  # Target top 10% of lags
    
    while time.time() - start_time < max_time:
        # Compute autoconvolution
        c = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(c)
        current_val = max_c * N / (N * N) # Sum is N, so s^2 = N^2
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Identify peaks
        # Use a threshold to select significant peaks
        # Or select top K indices
        K = int(N * peak_fraction)
        # c has length 2N-1
        # We want to reduce the peaks in c
        # Sort indices by value
        # To be fast, use partition
        # Get indices of largest K values
        # np.argpartition is faster than full sort
        part_idx = np.argpartition(c, -K)[-K:]
        peak_mask = np.zeros_like(c)
        peak_mask[part_idx] = 1.0
        
        # Compute gradient of sum of peaks w.r.t h
        # grad_h = sum_{m in Peaks} (h_{m-k} + h_{k-m})
        # This is 2 * cross_correlation(h, peak_mask)
        # cross_correlation(h, M) = convolve(h, flip(M))
        # Since M is symmetric? No, M is defined on c indices.
        # Let's use convolution properties.
        # c = h * h.
        # d(sum M*c)/dh = M * h + h * M (with appropriate alignment)
        
        # Convolution of M and h
        # M is length 2N-1, h is length N.
        # We need the part of the convolution that corresponds to h indices.
        
        # Let's compute g = convolve(peak_mask, current_h, mode='full')
        # The gradient component for h[k] is sum_{m} M[m] * (h[m-k] + h[k-m])
        # This is equivalent to the cross-correlation of h and M.
        
        # Efficient way:
        # grad = np.correlate(current_h, peak_mask, mode='full')
        # But we need to align it correctly.
        
        # Let's stick to the explicit sum but vectorized via FFT or simple conv?
        # N=600 is small enough that even O(N^2) might be okay if optimized, 
        # but let's try to be O(N log N) or O(N).
        
        # The term h[m-k] is non-zero when 0 <= m-k < N => k <= m < k+N.
        # The term h[k-m] is non-zero when 0 <= k-m < N => m <= k < m+N.
        
        # We can compute this by convolving h with the peak mask.
        # Let M be the peak mask (length 2N-1).
        # We want G[k] = sum_m M[m] * (h[m-k] + h[k-m])
        
        # Note that h[m-k] is like a shifted h.
        # Let's compute corr = np.correlate(current_h, peak_mask, mode='full')
        # The indices of corr are from -(N-1) to (2N-2) ? No.
        # np.correlate(a, v) computes sum a[i+k] * v[i].
        
        # Let's just use the property that the gradient of the L1 norm of peaks 
        # can be approximated by the convolution of the mask with h.
        
        # Simpler approach for N=600:
        # Just compute the gradient directly using broadcasting or loops?
        # N=600, K=60.
        # Loop over K peaks, add shifted h to gradient.
        # This is O(K*N) = 36000 ops. Very fast.
        
        G = np.zeros(N)
        for m in part_idx:
            # Add h[m-k] for valid k
            # m-k in [0, N-1] => k in [m-N+1, m]
            k_min = max(0, m - (N - 1))
            k_max = min(N - 1, m)
            if k_min <= k_max:
                G[k_min:k_max+1] += current_h[m - np.arange(k_min, k_max+1)]
            
            # Add h[k-m] for valid k
            # k-m in [0, N-1] => k in [m, m+N-1]
            k_min2 = max(0, m)
            k_max2 = min(N - 1, m + N - 1)
            if k_min2 <= k_max2:
                G[k_min2:k_max2+1] += current_h[np.arange(k_min2, k_max2+1) - m]
        
        # Normalize gradient
        G_mean = np.mean(G)
        G_std = np.std(G) + 1e-12
        G_norm = (G - G_mean) / G_std
        
        # Update h
        # Multiplicative update: h_new = h * (1 - alpha * G_norm)
        # Or additive: h_new = h - alpha * G
        # Multiplicative is often better for non-negativity.
        
        h_new = current_h * (1.0 - alpha * G_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            # If it died, restart with small perturbation
            h_new = current_h * (1.0 + 0.1 * np.random.randn(N))
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
        
        h_new = h_new * (N / s_new)
        
        # Check if improvement
        c_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(c_new) * N / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            current_val = val_new
            # Decay alpha slowly
            alpha *= 0.99
        else:
            # Reduce alpha
            alpha *= 0.9
            if alpha < 1e-4:
                break
                
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Try several random initializations
    rng = np.random.RandomState(42)
    
    for i in range(10):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Random initialization
        # Mix of uniform and gaussian
        if i % 2 == 0:
            init_f = rng.uniform(0.5, 1.5, N)
        else:
            init_f = np.abs(rng.randn(N)) + 0.1
        
        # Refine
        refined_f, refined_score = _local_refinement(
            init_f, start_time, DEADLINE - 5
        )
        
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
            
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {_compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
