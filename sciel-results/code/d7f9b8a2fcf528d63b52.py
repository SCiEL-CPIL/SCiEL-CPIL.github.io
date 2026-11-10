"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import random

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm(coeffs, num_samples=2048):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    # Dense sampling
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    # Evaluate P at all zs vectorized using Horner-like approach or direct powers
    # Direct powers approach for small n
    powers = np.ones((num_samples, n), dtype=complex)
    for k in range(1, n):
        powers[:, k] = powers[:, k-1] * zs
    vals = powers @ coeffs
    mag = np.abs(vals)
    
    # Find top peaks for refinement
    top_k = 5
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = 0.0
    for idx in peak_indices:
        center = angles[idx]
        # Golden-section search around the peak
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window), 
                              method='bounded', options={'xatol': 1e-10})
        if not res.success:
            continue
        peak_mag = -res.fun
        if peak_mag > best_val:
            best_val = peak_mag
    
    return best_val

def _score(coeffs, num_samples=2048):
    """Score: max |P(z)| / sqrt(n)"""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

def _score_fast(coeffs):
    """Faster score using fewer samples for local search steps."""
    return _sup_norm(coeffs, num_samples=512) / np.sqrt(len(coeffs))

def _greedy_local_search(coeffs, time_limit=30.0, fast=True):
    """Local search: try flipping single coefficients and pairs."""
    start_time = time.time()
    current = list(coeffs)
    score_func = _score_fast if fast else _score
    current_score = score_func(current)
    n = len(current)
    
    # Single flips
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            old_val = current[i]
            current[i] = -old_val
            new_score = score_func(current)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        # Pair flips
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, n):
                vi, vj = current[i], current[j]
                current[i] = -vi
                current[j] = -vj
                new_score = score_func(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = vi
                    current[j] = vj
    
    return current, current_score

def _random_restarts(num_restarts, time_per_restart):
    """Try multiple random initializations with local search."""
    best_coeffs = None
    best_score = float('inf')
    start_time = time.time()
    
    for r in range(num_restarts):
        if time.time() - start_time > time_per_restart:
            break
        # Random initialization
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _greedy_local_search(coeffs, time_limit=0.3, fast=True)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    return best_coeffs, best_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _perturb(coeffs, num_flips):
    """Create a perturbed version of coeffs by flipping random coefficients."""
    result = list(coeffs)
    for _ in range(num_flips):
        idx = random.randint(0, len(result) - 1)
        result[idx] = -result[idx]
    return result

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Start with Golay-Rudin-Shapiro
    grs = _grs_sequence(24)
    best_coeffs = list(grs)
    best_score = _score(best_coeffs)
    print(f"Initial GRS score: {best_score:.9f}")
    
    # Local search from GRS
    coeffs, score = _greedy_local_search(best_coeffs, time_limit=15.0, fast=True)
    if score < best_score:
        best_score = score
        best_coeffs = coeffs
        print(f"After LS from GRS: {best_score:.9f}")
    
    # Random restarts with local search
    if time.time() < deadline - 40:
        num_restarts = 100
        time_per_restart = min(50.0, deadline - time.time() - 20)
        coeffs, score = _random_restarts(num_restarts, time_per_restart)
        if coeffs is not None and score < best_score:
            best_score = score
            best_coeffs = coeffs
            print(f"After random restarts: {best_score:.9f}")
    
    # Perturbed local search from best
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() >= deadline - 15:
                break
            perturbed = _perturb(best_coeffs, random.randint(1, 4))
            coeffs, score = _greedy_local_search(perturbed, time_limit=1.0, fast=True)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # Final local search from best with full precision
    if time.time() < deadline - 10:
        coeffs, score = _greedy_local_search(best_coeffs, time_limit=min(10.0, deadline - time.time() - 5), fast=False)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Final score with high precision
    final_score = _score(best_coeffs)
    print(f"Final score: {final_score:.9f}")
    
    return best_coeffs
# EVOLVE-BLOCK-END
