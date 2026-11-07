"""
Spectral Gradient Descent for Erdos minimum overlap.

Strategy:
1. The objective is max_k corr_k / N * 2.
2. corr_k = sum_i h[i] * (1-h)[i+k].
3. This is a smooth function of h. We can compute the gradient of the active term (the one with max corr) efficiently using FFT or direct summation (N=100 is small enough for direct, but FFT is cleaner for 'spectral' framing).
4. Actually, for N=100, direct computation of the gradient of the max term is O(N^2) per step, which is very fast.
5. We use gradient descent with a projection step to maintain sum(h) = N/2 and 0 <= h <= 1.
6. To handle the 'max' non-differentiability, we use a soft-max approximation or simply follow the gradient of the current max term (which works well if the max is unique and stable).
7. We will use multiple restarts with different initial conditions (random, sinusoidal, block) to escape local minima.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0

def _score(h):
    """Compute the evaluation value (lower is better)."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    """Check if h is valid: finite, in [0,1], sum close to N/2."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """Project h onto the hyperplane sum(h) = N/2, then clip to [0,1]."""
    # Simple iterative projection
    h = h.copy()
    for _ in range(100):
        h = np.clip(h, 0, 1)
        diff = N / 2 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        # Distribute the difference uniformly among elements not at bounds
        # This is a rough projection; for better results, we can do a water-filling approach
        # But for gradient descent steps, a simple uniform adjustment often works if steps are small
        # However, uniform adjustment violates bounds if some are at 0 or 1.
        # Let's do a proper projection:
        # We want to find alpha such that sum(clip(h + alpha * mask)) = N/2
        # This is complex. Instead, we use a simpler heuristic for the gradient step:
        # Subtract the mean deviation.
        h = h + diff / N
        # Re-clip
        h = np.clip(h, 0, 1)
    # Final exact fix: if still off, adjust the most "central" elements
    diff = N / 2 - np.sum(h)
    if abs(diff) > 1e-9:
        # Find indices where we can adjust
        if diff > 0:
            # Need to increase sum
            candidates = np.where(h < 1.0)[0]
            # Sort by how much room they have? Or just pick random?
            # Let's pick the ones with the most room (1 - h[i])
            rooms = 1.0 - h[candidates]
            order = np.argsort(-rooms)
            idxs = candidates[order]
        else:
            # Need to decrease sum
            candidates = np.where(h > 0.0)[0]
            # Sort by how much they have
            vals = h[candidates]
            order = np.argsort(vals)
            idxs = candidates[order]
        
        remaining = abs(diff)
        for idx in idxs:
            if remaining <= 0:
                break
            if diff > 0:
                add = min(1.0 - h[idx], remaining)
                h[idx] += add
                remaining -= add
            else:
                sub = min(h[idx], -remaining)
                h[idx] -= sub
                remaining += sub
    return h

def _objective_and_grad(h):
    """
    Compute objective value and gradient.
    Objective: max_k corr_k / N * 2
    corr_k = sum_i h[i] * (1-h)[i+k]
    
    Let's compute all correlations.
    corr = np.correlate(h, 1-h, mode='full')
    max_idx = np.argmax(corr)
    k = max_idx - (N - 1)  # Lag index
    
    The term is T = sum_i h[i] * (1-h)[i+k]
    Note: indices must be valid.
    For a given lag k, the valid range of i is such that 0 <= i < N and 0 <= i+k < N.
    
    Gradient of T with respect to h[j]:
    T = sum_{i: i+k in [0,N)} h[i] (1 - h[i+k])
    
    Let's rewrite T = sum_{i in valid} (h[i] - h[i]h[i+k])
    
    dT/dh[j] = 
    1. If j is in the first index set (i): derivative of h[j] is (1 - h[j+k])
    2. If j is in the second index set (i+k): derivative of -h[i]h[j] is -h[j-k]
    
    So, dT/dh[j] = (1 - h[j+k]) * I[j+k valid] - h[j-k] * I[j-k valid]
    
    We need to be careful with the 'full' correlation definition.
    np.correlate(a, v, mode='full')
    c[k] = sum_n a[n+k] * v[n]
    Here a=h, v=1-h.
    c[k] = sum_n h[n+k] * (1-h[n])
    The index k ranges from -(N-1) to N-1.
    np.argmax returns index in [0, 2N-1].
    Let idx_max = np.argmax(corr).
    k_lag = idx_max - (N - 1).
    
    The term is c[k_lag] = sum_n h[n+k_lag] * (1-h[n])
    Valid n: 0 <= n < N and 0 <= n+k_lag < N.
    
    Let's compute the gradient vector g of length N, initialized to 0.
    For each valid n:
        i1 = n
        i2 = n + k_lag
        Term: h[i2] * (1 - h[i1])
        
        d/dh[i1]: h[i2] * (-1) = -h[i2]
        d/dh[i2]: (1 - h[i1]) * 1 = 1 - h[i1]
        
    So:
    g[i1] += -h[i2]
    g[i2] += 1 - h[i1]
    
    Objective value = c[k_lag] / N * 2
    Gradient of objective = g / N * 2
    """
    corr = np.correlate(h, 1 - h, mode="full")
    idx_max = np.argmax(corr)
    k_lag = idx_max - (N - 1)
    
    val = corr[idx_max] / N * 2
    
    grad = np.zeros(N)
    
    # Determine valid range for n
    # 0 <= n < N
    # 0 <= n + k_lag < N  => -k_lag <= n < N - k_lag
    
    n_start = max(0, -k_lag)
    n_end = min(N, N - k_lag)
    
    if n_start < n_end:
        ns = np.arange(n_start, n_end)
        i1 = ns
        i2 = ns + k_lag
        
        # g[i1] += -h[i2]
        grad[i1] += -h[i2]
        # g[i2] += 1 - h[i1]
        grad[i2] += 1 - h[i1]
        
    grad = grad / N * 2
    return val, grad

def _gradient_descent(h0, steps=1000, lr_init=0.01, lr_decay=0.999):
    """
    Perform gradient descent with projection.
    """
    h = h0.copy()
    h = _fix_sum(h)
    val, grad = _objective_and_grad(h)
    best_val = val
    best_h = h.copy()
    lr = lr_init
    
    for _ in range(steps):
        # Gradient step
        h_new = h - lr * grad
        
        # Project to feasible set
        h_new = _fix_sum(h_new)
        
        val_new, grad_new = _objective_and_grad(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        h = h_new
        grad = grad_new
        lr *= lr_decay
        
        # If lr gets too small, break
        if lr < 1e-8:
            break
            
    return best_h, best_val

def _sinusoidal_init(freq, phase):
    """Generate a sinusoidal candidate that satisfies sum constraint approximately."""
    h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
    h = np.clip(h, 0, 1)
    h = _fix_sum(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0
    
    # Initial best: seed
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Random uniform
    for _ in range(5):
        h = np.random.uniform(0, 1, N)
        h = _fix_sum(h)
        starts.append(h)
        
    # 2. Sinusoidal with different frequencies and phases
    for freq in [1, 2, 3, 5, 10]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h = _sinusoidal_init(freq, phase)
            starts.append(h)
            
    # 3. Block structures (alternating blocks)
    for block_len in [2, 5, 10, 25]:
        h = np.tile([0.0, 1.0], int(N / block_len))
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)), mode='edge')
        elif len(h) > N:
            h = h[:N]
        h = _fix_sum(h)
        starts.append(h)
        
    # 4. Random binary (balanced)
    for _ in range(5):
        h = np.zeros(N)
        ones = np.random.choice(N, size=N // 2, replace=False)
        h[ones] = 1.0
        starts.append(h)
        
    # Run gradient descent from each start
    for i, h0 in enumerate(starts):
        if time.time() > deadline - 2.0:
            break
            
        # Adaptive learning rate? Start with 0.01
        # If N=100, objective is O(1). Gradient is O(1).
        # Step size 0.01 is reasonable.
        
        h_opt, score_opt = _gradient_descent(h0, steps=500, lr_init=0.05, lr_decay=0.995)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
        # If we have time, do a second pass with smaller lr for refinement
        if time.time() < deadline - 5.0 and score_opt < best_score + 1e-4:
            h_opt2, score_opt2 = _gradient_descent(h_opt, steps=500, lr_init=0.01, lr_decay=0.999)
            if score_opt2 < best_score:
                best_score = score_opt2
                best_h = h_opt2.copy()
                
    # Final validation and fallback
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
