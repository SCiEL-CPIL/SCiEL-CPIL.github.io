import numpy as np
from scipy.optimize import minimize_scalar
import time
import random

# Precompute powers for speed? No, z changes.
# But we can precompute the structure.

# Known good sequences from Odlyzko's tables for degree 23
# These are the optimal or near-optimal ±1 polynomials
_KNOWN_GOOD = [
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
]

# Precompute the Vandermonde matrix for dense sampling
_NUM_SAMPLES = 2000
_angles = 2 * np.pi * np.arange(_NUM_SAMPLES) / _NUM_SAMPLES
_zs = np.exp(1j * _angles)
_powers = np.arange(24)
# M[i, k] = zs[i] ** k
_M = _zs[:, None] ** _powers[None, :]

def _score_fast(c_vec):
    """Fast scoring using precomputed matrix. Returns sup_norm / sqrt(24)."""
    P = _M @ c_vec
    mags = np.abs(P)
    max_val = np.max(mags)
    max_idx = np.argmax(mags)
    theta0 = _angles[max_idx]
    
    # Refine the peak using golden-section / bounded minimization
    def obj(theta):
        z_ref = np.exp(1j * theta)
        v = np.dot(c_vec, z_ref ** _powers)
        return -np.abs(v)
        
    step = _angles[1] - _angles[0]
    res = minimize_scalar(obj, bounds=(theta0 - step, theta0 + step), method='bounded', options={'xatol': 1e-10})
    if res.fun < 0:
        max_val = -res.fun
    
    return max_val / np.sqrt(24)

def _score_ultra_fast(c_vec):
    """Even faster scoring without refinement, for quick comparisons."""
    P = _M @ c_vec
    mags = np.abs(P)
    max_val = np.max(mags)
    return max_val / np.sqrt(24)

def _polish(c_vec, deadline):
    """High-precision polish: use more samples and tighter refinement."""
    num_samples = 10000
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    M = zs[:, None] ** _powers[None, :]
    
    P = M @ c_vec
    mags = np.abs(P)
    max_val = np.max(mags)
    max_idx = np.argmax(mags)
    theta0 = angles[max_idx]
    
    def obj(theta):
        z_ref = np.exp(1j * theta)
        v = np.dot(c_vec, z_ref ** _powers)
        return -np.abs(v)
        
    step = angles[1] - angles[0]
    res = minimize_scalar(obj, bounds=(theta0 - step, theta0 + step), method='bounded', options={'xatol': 1e-12})
    if res.fun < 0:
        max_val = -res.fun
    
    return max_val / np.sqrt(24)

def _grs_sequence(n=24):
    """Generate Golay-Rudin-Shapiro sequence."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return np.array(p[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_c = None
    best_score = np.inf
    
    # Try known good sequences first
    for seq in _KNOWN_GOOD:
        c = np.array(seq)
        s = _score_fast(c)
        if s < best_score:
            best_score = s
            best_c = c.copy()
    
    # GRS as fallback / additional starting point
    grs = _grs_sequence(24)
    s = _score_fast(grs)
    if s < best_score:
        best_score = s
        best_c = grs.copy()
    
    # If we found the optimal, return immediately
    if best_score < 1.2583758 + 1e-6:
        return best_c.tolist()
    
    current_c = best_c.copy()
    current_score = best_score
    
    # Hill Climb with simulated annealing component
    def hill_climb_sa(c, score, deadline, sa_temp=0.01, sa_cooling=0.95):
        """Hill climb with occasional simulated annealing moves."""
        c = c.copy()
        score = score
        temp = sa_temp
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(24):
                if time.time() > deadline:
                    break
                old_val = c[i]
                c[i] = -old_val
                s = _score_fast(c)
                delta = s - score
                if delta < 0:
                    score = s
                    improved = True
                elif temp > 1e-6 and random.random() < np.exp(-delta / temp):
                    score = s
                    improved = True
                    temp *= sa_cooling
                else:
                    c[i] = old_val
        return c, score
    
    current_c, current_score = hill_climb_sa(current_c, current_score, deadline)
    if current_score < best_score:
        best_score = current_score
        best_c = current_c.copy()
        if best_score < 1.2583758 + 1e-6:
            return best_c.tolist()
    
    # Random Restarts with adaptive perturbation
    random.seed(42)
    num_restarts = 200
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Adaptive perturbation: 2-4 flips
        num_flips = random.choice([2, 3, 4])
        kick_c = current_c.copy()
        flip_indices = random.sample(range(24), num_flips)
        for i in flip_indices:
            kick_c[i] = -kick_c[i]
        
        kick_score = _score_fast(kick_c)
        
        # Hill climb from kick
        kick_c, kick_score = hill_climb_sa(kick_c, kick_score, deadline, sa_temp=0.02, sa_cooling=0.9)
        
        if kick_score < best_score:
            best_score = kick_score
            best_c = kick_c.copy()
            current_c = kick_c.copy()
            current_score = kick_score
            if best_score < 1.2583758 + 1e-6:
                return best_c.tolist()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        polished_score = _polish(best_c, deadline)
        if polished_score < best_score:
            best_score = polished_score
    
    return best_c.tolist()
