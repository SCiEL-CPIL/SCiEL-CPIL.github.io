import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_array(h):
    """Return array of overlap values for all shifts k=1..n-1."""
    n = len(h)
    ovs = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.zeros(n)
        h_shifted[:n - k] = h[k:]
        ovs[k - 1] = np.sum(h * (1.0 - h_shifted)) * (2.0 / n)
    return ovs


def _overlap_max(h):
    return _overlap_array(h).max()


def construct():
    deadline = time.time() + 115.0
    n = N
    half = n / 2.0
    
    # Initialize with a smooth bump (high center, zero edges)
    x = (np.arange(n) + 0.5) * 2.0 / n
    bump = np.cos(np.pi * (x - 1.0) / 2.0) ** 2
    h = bump / bump.sum() * half
    h = np.clip(h, 0.0, 1.0)
    h += (half - h.sum()) / n
    h = np.clip(h, 0.0, 1.0)
    
    best_h = h.copy()
    best_score = _overlap_max(h)
    
    # Local search with adaptive step size
    step = 0.03
    no_improve = 0
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            # Perturb random coordinates
            n_pert = rng.randint(3, 10)
            idx = rng.randint(0, n, size=n_pert)
            old = h[idx].copy()
            h[idx] += rng.randn(n_pert) * step
            h[idx] = np.clip(h[idx], 0.0, 1.0)
            # Fix sum
            diff = half - h.sum()
            h += diff / n
            h = np.clip(h, 0.0, 1.0)
            
            sc = _overlap_max(h)
            if sc < best_score - 1e-10:
                best_score = sc
                best_h = h.copy()
                improved = True
                no_improve = 0
            else:
                h[idx] = old
                diff = half - h.sum()
                h += diff / n
                h = np.clip(h, 0.0, 1.0)
        
        if not improved:
            no_improve += 1
            step *= 0.9
            if step < 1e-5:
                break
        else:
            step = min(step * 1.05, 0.1)
    
    # Final cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = half - best_h.sum()
    best_h += diff / n
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
