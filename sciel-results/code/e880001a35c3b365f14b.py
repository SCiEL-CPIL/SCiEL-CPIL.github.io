"""Erdos minimum overlap - numerical optimization approach."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140  # seconds, safety margin before 150s kill

def _compute_overlaps(h):
    """Compute overlap of h with shifted (1-h) for all non-zero shifts."""
    # overlap[k] = sum_i h[i] * (1 - h[(i-k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
    # = N/2 - autocorr[k]  (when sum(h) = N/2)
    h = np.asarray(h)
    n = len(h)
    overlaps = np.empty(n - 1)
    for k in range(1, n):
        shifted = np.roll(h, k)
        overlaps[k - 1] = np.sum(h * (1.0 - shifted))
    return overlaps

def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    # Fine-tune to make sum exactly N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute the difference
        if diff > 0:
            idx = np.argsort(-h)
            for i in idx:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
                if diff <= 0:
                    break
        else:
            idx = np.argsort(h)
            for i in idx:
                rem = min(-diff, h[i])
                h[i] -= rem
                diff += rem
                if diff >= 0:
                    break
    return h

def construct():
    start_time = time.time()
    best_h = np.full(N, 0.5)
    best_score = np.max(_compute_overlaps(best_h))
    
    def objective(x):
        h = _project(x)
        overlaps = _compute_overlaps(h)
        return np.max(overlaps)
    
    rng = np.random.default_rng(42)
    
    # Strategy: random restarts with Nelder-Mead
    # Use time budget to determine number of restarts
    while time.time() - start_time < DEADLINE - 10:
        # Generate random initial point
        x0 = rng.random(N)
        x0 = _project(x0)
        
        # Also try some structured initializations
        if rng.random() < 0.3:
            # Alternating pattern with noise
            x0 = np.where(np.arange(N) % 2 == 0, 0.8, 0.2) + rng.normal(0, 0.1, N)
        elif rng.random() < 0.5:
            # Smooth random
            x0 = np.cumsum(rng.random(N))
            x0 = x0 / x0[-1] * N / 2.0
        
        x0 = _project(x0)
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-7,
                    'adaptive': True,
                }
            )
        except Exception:
            continue
        
        h_opt = _project(result.x)
        score = np.max(_compute_overlaps(h_opt))
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Check deadline
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final projection to ensure exact constraints
    best_h = _project(best_h)
    # Verify sum is exactly N/2
    s = np.sum(best_h)
    if abs(s - N/2) > 1e-10:
        best_h = best_h + (N/2 - s) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
