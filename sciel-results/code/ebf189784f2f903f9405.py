import numpy as np
import time

N = 600
H = 1.0 / N  # interval width

def _autoconv_peak(h):
    """Compute the maximum value of the autoconvolution of step heights h.
    The autoconvolution on the continuous domain has peaks at the interval midpoints.
    For a step function with heights h on equal intervals of width H,
    the autoconvolution at the center of the k-th output interval is:
        sum_i h[i] * h[k-i] * (overlap width)
    But more precisely, the convolution of two piecewise-constant functions
    on equal intervals: the result at position t is an integral.
    
    For discrete step functions with N equal bins, the autoconvolution evaluated
    at the bin centers gives:
        conv[k] = H * sum_i h[i] * h[k - i]
    where we use linear convolution with appropriate padding.
    
    Actually, the normalized maximum: the task says "normalized maximum of its autoconvolution."
    The autoconvolution of a function f with itself is (f * f)(x) = ∫ f(y)f(x-y) dy.
    For a step function on [-1/4, 1/4] with N intervals of width H, the autoconvolution
    is supported on [-1/2, 1/2]. The maximum of this autoconvolution is what we want to minimize,
    and "normalized" likely means divided by (∫f)^2 or similar.
    
    Let's think: if f is a step function with heights h[0..N-1] on intervals of width H,
    then ∫f = H * sum(h). The autoconvolution max is what we measure.
    
    The "normalized maximum" is likely: max(f*f) / (∫f)^2 * (1/width) or something.
    But let's just compute max of the discrete autoconvolution and normalize appropriately.
    
    For the discrete case: the autoconvolution at the center of interval k (in the output)
    is approximately H * sum_i h[i] * h[k-i] for the linear convolution.
    
    The output has 2N-1 intervals. Let's compute the full linear convolution.
    """
    # Linear convolution of h with itself, scaled by H for the integral
    conv = np.convolve(h, h, mode='full') * H
    # The maximum of the autoconvolution
    return np.max(conv)

def _normalize_peak(h):
    """Compute the normalized maximum of the autoconvolution.
    Normalization: divide by (integral of f)^2 to make it scale-invariant.
    integral = H * sum(h)
    """
    integral = H * np.sum(h)
    if integral <= 0:
        return float('inf')
    peak = _autoconv_peak(h)
    return peak / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Start with a reasonable initial guess: a triangular/bell-shaped profile
    # which tends to have lower peak autoconvolution than uniform.
    
    # Initialize with a smooth bell curve
    x = np.linspace(-1.0, 1.0, N)
    sigma = 0.6
    h = np.exp(-0.5 * (x / sigma) ** 2)
    h = np.maximum(h, 0.0)
    
    best_h = h.copy()
    best_score = _normalize_peak(h)
    
    # Iterative optimization using projected gradient / multiplicative updates
    # Minimize the peak of the autoconvolution subject to nonnegativity.
    
    # Use a subgradient approach: the peak of the autoconvolution is a function
    # of the convolution at the maximizing index. We can use multiplicative updates
    # (similar to IPF) to reduce the peak.
    
    lr = 0.5
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        conv = np.convolve(h, h, mode='full')
        max_idx = np.argmax(conv)
        peak_val = conv[max_idx]
        
        if peak_val <= 1e-15:
            break
        
        # Compute the gradient of the peak w.r.t. h
        # conv[max_idx] = sum_i h[i] * h[max_idx - i]
        # d(conv[max_idx])/d(h[j]) = h[max_idx - j] + h[j - max_idx]  (where valid)
        
        grad = np.zeros(N)
        for j in range(N):
            i1 = max_idx - j
            if 0 <= i1 < N:
                grad[j] += h[i1]
            i2 = j - max_idx
            if 0 <= i2 < N:
                grad[j] += h[i2]
        
        # Multiplicative update to reduce the peak
        # h_new[j] = h[j] * exp(-lr * grad[j] / peak_val)
        # But this can be unstable. Use a softer update.
        
        # Actually, let's use a simpler approach: scale down elements that contribute
        # most to the peak convolution.
        
        # The contribution of h[j] to conv[max_idx] is h[j] * (sum of paired h values)
        # Let's just do a simple gradient descent with projection.
        
        h_new = h - lr * grad / peak_val
        h_new = np.maximum(h_new, 0.0)
        
        # Preserve the integral to avoid trivial solutions
        old_int = np.sum(h)
        new_int = np.sum(h_new)
        if new_int > 1e-15:
            h_new = h_new * (old_int / new_int)
        
        new_score = _normalize_peak(h_new)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
        
        h = h_new
        
        # Anneal learning rate
        lr = max(0.01, 0.5 * (0.98 ** iteration))
    
    # Try a few different initializations
    for sigma in [0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2]:
        if time.time() > deadline - 5.0:
            break
        
        x = np.linspace(-1.0, 1.0, N)
        h = np.exp(-0.5 * (x / sigma) ** 2)
        h = np.maximum(h, 0.0)
        
        for iteration in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            max_idx = np.argmax(conv)
            peak_val = conv[max_idx]
            
            if peak_val <= 1e-15:
                break
            
            grad = np.zeros(N)
            for j in range(N):
                i1 = max_idx - j
                if 0 <= i1 < N:
                    grad[j] += h[i1]
                i2 = j - max_idx
                if 0 <= i2 < N:
                    grad[j] += h[i2]
            
            h_new = h - lr * grad / peak_val
            h_new = np.maximum(h_new, 0.0)
            
            old_int = np.sum(h)
            new_int = np.sum(h_new)
            if new_int > 1e-15:
                h_new = h_new * (old_int / new_int)
            
            new_score = _normalize_peak(h_new)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            
            h = h_new
        
        lr = max(0.01, 0.5 * (0.98 ** 2000))
    
    # Final: try a few random perturbations of the best solution
    rng = np.random.default_rng(42)
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        
        h = best_h + rng.normal(0, 0.05, size=N)
        h = np.maximum(h, 0.0)
        
        for iteration in range(500):
            if time.time() > deadline - 2.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            max_idx = np.argmax(conv)
            peak_val = conv[max_idx]
            
            if peak_val <= 1e-15:
                break
            
            grad = np.zeros(N)
            for j in range(N):
                i1 = max_idx - j
                if 0 <= i1 < N:
                    grad[j] += h[i1]
                i2 = j - max_idx
                if 0 <= i2 < N:
                    grad[j] += h[i2]
            
            h_new = h - 0.3 * grad / peak_val
            h_new = np.maximum(h_new, 0.0)
            
            old_int = np.sum(h)
            new_int = np.sum(h_new)
            if new_int > 1e-15:
                h_new = h_new * (old_int / new_int)
            
            new_score = _normalize_peak(h_new)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            
            h = h_new
    
    # Return the best solution as a list of floats
    return [float(x) for x in best_h]
