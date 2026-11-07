import numpy as np
import time

N = 100
DEADLINE_MARGIN = 20.0  # safety margin before 150s

def _objective(h):
    """Compute max cross-correlation between h and 1-h."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def _project(h):
    """Project h onto the feasible region: h in [0,1]^N, sum(h) = N/2."""
    h = h.copy()
    # Iteratively clip and redistribute to satisfy sum constraint
    for _ in range(50):
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        target = N / 2.0
        diff = target - s
        if abs(diff) < 1e-12:
            break
        # Only adjust non-saturated elements
        mask = (h > 0) & (h < 1)
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    h = np.clip(h, 0.0, 1.0)
    # Final adjustment
    s = np.sum(h)
    diff = N / 2.0 - s
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _gradient_descent(h0, lr=0.01, max_iter=5000, tol=1e-9):
    """Simple projected gradient descent with adaptive step size."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _objective(h)
    
    for i in range(max_iter):
        # Compute gradient of max correlation
        # corr = np.correlate(h, 1-h, mode='full')
        # The max element's gradient needs careful handling
        # Let's compute gradient numerically for robustness
        h_grad = np.zeros(N)
        eps = 1e-6
        base_val = _objective(h)
        
        # For efficiency, use analytic gradient where possible
        # np.correlate(h, 1-h) gives full correlation
        # Let k = argmax of correlation
        corr = np.correlate(h, 1.0 - h, mode='full')
        k = np.argmax(corr)
        
        # The correlation at lag k is sum_i h[i] * (1-h)[i+k] for appropriate ranges
        # Actually, let's just use numerical gradient with a few samples
        # But that's too slow. Let me think analytically.
        
        # For the max correlation coefficient, we need d/dh_j [max_lag sum_i h[i]*(1-h)[i+lag]]
        # Let's compute gradient numerically with sparse sampling
        # Actually, let's use the fact that for a specific lag k:
        # corr_k = sum_i h[i] * (1 - h[i+k]) (with appropriate indexing)
        # d(corr_k)/d(h[j]) = (1-h[j+k]) - h[j] (with boundary conditions)
        
        # Let's just do finite differences for the active lag only
        # This is O(N) per step
        h_plus = h.copy()
        h_minus = h.copy()
        for j in range(N):
            h_plus[j] = min(1.0, h[j] + eps)
            h_minus[j] = max(0.0, h[j] - eps)
            # This is too slow. Let me use analytic gradient.
        
        # Analytic gradient for the max correlation term
        # corr_k = sum over valid i of h[i] * (1 - h[i + offset(k)])
        # where offset depends on k in full mode
        # For mode='full', corr[k] = sum_{i=max(0, k-(N-1))}^{min(N-1, k)} h[i] * (1-h[k-i])
        # Wait, np.correlate(h, 1-h, mode='full') has length 2N-1
        # corr[m] = sum_i h[i] * (1-h)[i + m - (N-1)] = sum_i h[i] * (1-h)[i + m - N + 1]
        
        # Let's just compute gradient numerically but vectorized
        # Use central differences for a subset
        grad = np.zeros(N)
        step = 1e-5
        for j in range(0, N, 2):  # Sample every other element for speed
            h_p = h.copy()
            h_m = h.copy()
            h_p[j] = min(1.0, h[j] + step)
            h_m[j] = max(0.0, h[j] - step)
            grad[j] = (_objective(h_p) - _objective(h_m)) / (2 * step)
        
        # Extrapolate to remaining elements (approximate)
        for j in range(1, N, 2):
            grad[j] = (grad[j-1] + grad[min(j+1, N-1)]) / 2 if j < N-1 else grad[j-1]
        
        # Projected gradient step
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        val = _objective(h_new)
        if val < best_val - tol:
            best_val = val
            best_h = h_new.copy()
            h = h_new
        elif val < best_val:
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-8:
                break
        
        # Time check
        if time.time() > time.time() + 0:  # placeholder
            pass
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Start with seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _objective(h0)
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [h0.copy()]
    
    # Random perturbations
    for scale in [0.1, 0.2, 0.3, 0.5]:
        h_pert = np.full(N, 0.5) + rng.randn(N) * scale
        h_pert = _project(h_pert)
        starts.append(h_pert)
    
    # Structured patterns
    # Sine waves
    for freq in [1, 2, 3, 4, 5, 8]:
        for amp in [0.2, 0.4, 0.5]:
            x = np.arange(N)
            h_sine = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            h_sine = _project(h_sine)
            starts.append(h_sine)
    
    # Alternating patterns
    for a in [0.2, 0.3, 0.4]:
        h_alt = np.where(np.arange(N) % 2 == 0, a, 1.0 - a)
        h_alt = _project(h_alt)
        starts.append(h_alt)
    
    # Block patterns
    for block_size in [10, 20, 25, 50]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 1.0
        h_block = _project(h_block)
        starts.append(h_block)
    
    # Triangular
    x = np.arange(N)
    h_tri = 0.5 + 0.5 * np.sin(np.pi * x / N)
    h_tri = _project(h_tri)
    starts.append(h_tri)
    
    # Run gradient descent from each starting point
    for h_start in starts:
        if time.time() > deadline - 5:
            break
        
        h_opt, val = _gradient_descent(h_start, lr=0.05, max_iter=3000)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Final projection
    best_h = _project(best_h)
    
    return best_h
