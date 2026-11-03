"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    h = np.asarray(h)
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_from_params(params):
    """Construct h from frequency, amplitude, phase."""
    f, a, phi = params
    i = np.arange(N)
    h = 0.5 + a * np.cos(2.0 * np.pi * f * i / N + phi)
    # Clip to [0,1] and renormalize to sum = N/2
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0.0, 1.0)
    # Final adjustment to ensure exact sum
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try a grid of frequencies and phases
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try different frequencies (integer and non-integer)
    freqs = list(range(1, N // 2 + 1)) + [N * 0.1, N * 0.2, N * 0.3, N * 0.4, N * 0.7]
    
    for f in freqs:
        if time.time() > deadline:
            break
        for a in [0.1, 0.2, 0.3, 0.4, 0.45]:
            if time.time() > deadline:
                break
            for phi in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                if time.time() > deadline:
                    break
                h = _construct_from_params([f, a, phi])
                score = _compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try local optimization on the best found
    if time.time() < deadline - 10:
        def objective(params):
            h = _construct_from_params(params)
            return _compute_max_overlap(h)
        
        # Start from best found
        best_f = 1.0
        best_a = 0.3
        best_phi = 0.0
        
        # Try to find good initial params by checking what best_h looks like
        # For now, try a few starting points
        for init_f in [1, 2, 3, N * 0.1, N * 0.3]:
            if time.time() > deadline - 5:
                break
            for init_a in [0.2, 0.3, 0.4]:
                if time.time() > deadline - 5:
                    break
                for init_phi in [0, np.pi / 2, np.pi]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        res = minimize(objective, [init_f, init_a, init_phi],
                                      method='Nelder-Mead',
                                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
                        if res.fun < best_score:
                            best_score = res.fun
                            best_h = _construct_from_params(res.x)
                    except Exception:
                        pass
    
    return best_h
# EVOLVE-BLOCK-END
