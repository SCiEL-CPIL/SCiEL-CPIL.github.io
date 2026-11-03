"""Erdos minimum overlap - frequency-domain optimization approach."""
import numpy as np
import time

N = 100
DEADLINE = 110.0  # safety margin before 120s limit


def _compute_overlaps(h):
    """Compute overlap at each shift k: sum_i h[i]*(1-h[(i+k)%N])"""
    # overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - autocorr[k]  (since sum h = N/2)
    autocorr = np.zeros(N)
    for k in range(N):
        autocorr[k] = np.dot(h, np.roll(h, k))
    overlaps = N / 2.0 - autocorr
    return overlaps


def _max_overlap(h):
    """Return the maximum overlap value."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)


def _project(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    target = N / 2.0
    current_sum = np.sum(h)
    if current_sum != target:
        diff = target - current_sum
        # Distribute adjustment to elements not at bounds
        mask = (h > 0.0) & (h < 1.0)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Fine-tune with small elements
            residual = target - np.sum(h)
            if abs(residual) > 1e-12:
                # Adjust the element closest to 0.5
                idx = np.argmin(np.abs(h - 0.5))
                h[idx] += residual
                h[idx] = np.clip(h[idx], 0.0, 1.0)
    return h


def construct():
    start_time = time.time()
    
    # Initial guess: sinusoidal pattern to create periodic structure
    # A sinusoid with period p will have high autocorrelation at shift p
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try various sinusoidal initializations
    for freq in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        t = np.arange(N)
        # h[i] = 0.5 + A * cos(2*pi*freq*i/N)
        # Need sum = N/2, which is satisfied since cos has zero mean
        # Need 0 <= h <= 1, so A <= 0.5
        for A in [0.1, 0.2, 0.3, 0.4, 0.49]:
            if time.time() - start_time > DEADLINE:
                break
            h = 0.5 + A * np.cos(2 * np.pi * freq * t / N)
            h = _project(h)
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Subgradient descent to further improve
    # The objective is max_k (N/2 - autocorr[k])
    # Subgradient w.r.t. h: for the maximizing shift k*, 
    # d(overlap[k*])/dh[i] = 1 - h[(i+k*)%N]
    
    h = best_h.copy()
    lr = 0.05
    for iteration in range(5000):
        if time.time() - start_time > DEADLINE:
            break
        
        overlaps = _compute_overlaps(h)
        max_idx = np.argmax(overlaps)
        max_val = overlaps[max_idx]
        
        # Subgradient of overlap[max_idx] w.r.t. h[i] = 1 - h[(i+max_idx)%N]
        grad = 1.0 - np.roll(h, max_idx)
        
        h = h - lr * grad
        h = _project(h)
        
        new_score = _max_overlap(h)
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    # Final refinement with smaller steps
    lr = 0.01
    for iteration in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        
        overlaps = _compute_overlaps(h)
        # Use top few shifts for more stable gradient
        top_k = np.argsort(overlaps)[-5:]
        grad = np.zeros(N)
        for k in top_k:
            grad += 1.0 - np.roll(h, k)
        grad /= len(top_k)
        
        h = h - lr * grad
        h = _project(h)
        
        new_score = _max_overlap(h)
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
    
    # Ensure exact sum
    best_h = _project(best_h)
    
    return best_h


# Verify at module level that construct works
if __name__ == "__main__":
    h = construct()
    print(f"Sum: {np.sum(h)}, Max overlap: {_max_overlap(h)}")
