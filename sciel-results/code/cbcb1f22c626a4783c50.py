"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_overlap(h):
    """Compute the maximum overlap between h and shifted (1-h).
    Overlap at shift s is sum_i min(h[i], 1-h[(i+s)%N]) / N
    """
    comp = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted_comp = np.roll(comp, s)
        ov = np.sum(np.minimum(h, shifted_comp)) / N
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _try_sinusoidal(k, a, phase=0.0):
    """Try h[i] = 0.5 + a*cos(2*pi*k*i/N + phase)"""
    i = np.arange(N)
    h = 0.5 + a * np.cos(2 * np.pi * k * i / N + phase)
    # Clamp to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Adjust to ensure sum = N/2
    # If clamping changed the sum, rescale the non-clamped part
    if not np.allclose(h.sum(), N / 2.0):
        # Try to fix by adjusting
        diff = N / 2.0 - h.sum()
        # Distribute adjustment
        mask = (h > 0.0) & (h < 1.0)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
            # If still off, accept small error or try different params
            if abs(h.sum() - N / 2.0) > 1e-6:
                return None, 1.0
    return h, _compute_max_overlap(h)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)  # Should be ~0.5
    
    # Try various sinusoidal perturbations
    best_found = best_score
    best_h_found = best_h.copy()
    
    for k in range(1, N // 2 + 1):
        if time.time() > deadline - 5.0:
            break
        for a in np.linspace(0.01, 0.49, 20):
            if time.time() > deadline - 5.0:
                break
            for phase in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                if time.time() > deadline - 5.0:
                    break
                h, score = _try_sinusoidal(k, a, phase)
                if h is not None and score < best_found:
                    best_found = score
                    best_h_found = h.copy()
    
    # If we found something better, use it
    if best_found < best_score:
        return best_h_found
    else:
        return best_h
# EVOLVE-BLOCK-END
