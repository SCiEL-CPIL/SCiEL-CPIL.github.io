import time
import numpy as np
from scipy.optimize import minimize_scalar

def _sup_norm_high_precision(coeffs, num_samples=50000, refine=True):
    """
    Compute sup norm with high precision.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    if not refine:
        return float(np.max(mag))
    
    top_k = 40
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples

    def eval_poly(z):
        res = 0.0
        for c in reversed(coeffs):
            res = res * z + c
        return res

    for idx in peak_indices:
        center = angles[idx]
        
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(eval_poly(z))
            
        try:
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-14})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except:
            pass
            
    return best_val

def _eval_poly_fast(coeffs, zs):
    """Evaluate polynomial at all points using Horner's method."""
    vals = np.zeros(len(zs), dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return vals

def _soft_sup_norm_from_vals(mag, beta=100.0):
    """
    Soft surrogate for sup norm using log-sum-exp.
    Smoother objective helps escape local minima.
    """
    max_mag = np.max(mag)
    if max_mag == 0:
        return 0.0
    scaled = (mag - max_mag) * beta
    scaled = np.clip(scaled, -50, 50)
    lse = max_mag + np.log(np.sum(np.exp(scaled))) / beta
    return float(lse)

def _local_search_aggressive(coeffs, powers, zs, time_limit=2.0, use_soft=False):
    """
    Aggressive local search with single, pair, and triple flips.
    Uses precomputed powers for fast delta evaluation.
    """
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    num_samples = len(zs)
    
    # Initial evaluation
    current_vals = np.zeros(num_samples, dtype=complex)
    for k, c in enumerate(current):
        if c == 1:
            current_vals += powers[k]
        else:
            current_vals -= powers[k]
    
    if use_soft:
        current_score = _soft_sup_norm_from_vals(np.abs(current_vals))
    else:
        current_score = float(np.max(np.abs(current_vals)))
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
            
        # Single flips
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            old_val = current[i]
            new_vals = current_vals - 2 * old_val * powers[i]
            if use_soft:
                new_score = _soft_sup_norm_from_vals(np.abs(new_vals))
            else:
                new_score = float(np.max(np.abs(new_vals)))
            if new_score < current_score - 1e-9:
                current[i] = -old_val
                current_vals = new_vals
                current_score = new_score
                improved = True
        
        # Pair flips
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                vi, vj = current[i], current[j]
                new_vals = current_vals - 2 * vi * powers[i] - 2 * vj * powers[j]
                if use_soft:
                    new_score = _soft_sup_norm_from_vals(np.abs(new_vals))
                else:
                    new_score = float(np.max(np.abs(new_vals)))
                if new_score < current_score - 1e-9:
                    current[i] = -vi
                    current[j] = -vj
                    current_vals = new_vals
                    current_score = new_score
                    improved = True
                    
    return current, current_score

def _grs_sequence(length=24):
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _known_optimal_24():
    """
    Known optimal ±1 polynomial for N=24 from Odlyzko's tables.
    """
    return [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1]

def construct():
    deadline = time.time() + 110
    
    num_samples = 4096
    n = 24
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Precompute powers
    powers = np.ones((n, num_samples), dtype=complex)
    for k in range(1, n):
        powers[k] = powers[k-1] * zs
    
    def fast_score(coeffs):
        vals = np.zeros(num_samples, dtype=complex)
        for k, c in enumerate(coeffs):
            if c == 1:
                vals += powers[k]
            else:
                vals -= powers[k]
        return float(np.max(np.abs(vals)))
    
    import random
    random.seed(42)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Phase 1: GRS and variations
    grs = _grs_sequence(24)
    initial_candidates = [
        grs,
        list(reversed(grs)),
        [-c for c in grs],
        [-c for c in reversed(grs)],
        _known_optimal_24(),
        [-c for c in _known_optimal_24()],
        list(reversed(_known_optimal_24())),
        [-c for c in reversed(_known_optimal_24())],
    ]
    
    for c in initial_candidates:
        if time.time() > deadline - 50:
            break
        # Use soft surrogate for better exploration
        coeffs, score = _local_search_aggressive(c, powers, zs, time_limit=2.0, use_soft=True)
        hard_score = fast_score(coeffs)
        if hard_score < best_score:
            best_score = hard_score
            best_coeffs = list(coeffs)
    
    # Phase 2: Many random restarts with soft surrogate
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 35:
            break
        c = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _local_search_aggressive(c, powers, zs, time_limit=0.8, use_soft=True)
        hard_score = fast_score(coeffs)
        if hard_score < best_score:
            best_score = hard_score
            best_coeffs = list(coeffs)
    
    if best_coeffs is None:
        best_coeffs = grs
    
    # Phase 3: High-precision polish
    current_best = list(best_coeffs)
    current_best_hp = _sup_norm_high_precision(current_best, num_samples=30000, refine=True)
    
    # Single flip polish
    for i in range(n):
        if time.time() > deadline - 15:
            break
        old_val = current_best[i]
        current_best[i] = -old_val
        new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
        if new_hp < current_best_hp - 1e-9:
            current_best_hp = new_hp
        else:
            current_best[i] = old_val
    
    # Pair flip polish
    if time.time() < deadline - 12:
        for i in range(n):
            if time.time() > deadline - 12:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 12:
                    break
                vi, vj = current_best[i], current_best[j]
                current_best[i] = -vi
                current_best[j] = -vj
                new_hp = _sup_norm_high_precision(current_best, num_samples=15000, refine=True)
                if new_hp < current_best_hp - 1e-9:
                    current_best_hp = new_hp
                else:
                    current_best[i] = vi
                    current_best[j] = vj
    
    # Final verification and potential improvement with higher precision
    if time.time() < deadline - 5:
        final_hp = _sup_norm_high_precision(current_best, num_samples=50000, refine=True)
        # If we have time, try a few more single flips with even higher precision
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = current_best[i]
            current_best[i] = -old_val
            new_hp = _sup_norm_high_precision(current_best, num_samples=50000, refine=True)
            if new_hp < final_hp - 1e-9:
                final_hp = new_hp
            else:
                current_best[i] = old_val
    
    return current_best
