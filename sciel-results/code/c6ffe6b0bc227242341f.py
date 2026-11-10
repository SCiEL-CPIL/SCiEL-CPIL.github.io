"""Optimal +/-1 polynomial with 24 coefficients minimizing sup norm on |z|=1."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import random
import itertools
import math

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm(coeffs, num_samples=4096, refine=True):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Vectorized evaluation using Horner's method
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    
    mag = np.abs(vals)
    
    if not refine:
        return float(np.max(mag))
    
    # Find top peaks for refinement
    top_k = 8
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    for idx in peak_indices:
        center = angles[idx]
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window),
                              method='bounded', options={'xatol': 1e-11})
        if res.success:
            peak_mag = -res.fun
            if peak_mag > best_val:
                best_val = peak_mag
    
    return best_val

def _score(coeffs, num_samples=4096):
    """Score: max |P(z)| / sqrt(n)"""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

_GLOBAL_ZS_1024 = None

def _get_zs_1024():
    global _GLOBAL_ZS_1024
    if _GLOBAL_ZS_1024 is None:
        angles = 2 * np.pi * np.arange(1024) / 1024
        _GLOBAL_ZS_1024 = np.exp(1j * angles)
    return _GLOBAL_ZS_1024

def _score_fast(coeffs, num_samples=1024):
    """Faster score for local search: dense sampling without refinement."""
    n = len(coeffs)
    zs = _get_zs_1024()
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)

def _local_search(coeffs, time_limit=5.0, fast=True):
    """Coordinate descent local search. Flips single bits if it improves the score."""
    start = time.time()
    current = list(coeffs)
    n = len(current)
    score_func = _score_fast if fast else _score
    current_score = score_func(current)
    improved = True
    
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            current[i] *= -1
            new_score = score_func(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] *= -1
    return current, current_score

def _simulated_annealing(initial_coeffs, time_limit=5.0, T_start=0.05, T_end=0.001, alpha=0.9995):
    """Simulated Annealing for escaping local minima."""
    start = time.time()
    current = list(initial_coeffs)
    n = len(current)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    T = T_start
    
    while time.time() - start < time_limit:
        # Generate neighbor
        idx = random.randint(0, n-1)
        current[idx] *= -1
        new_score = _score_fast(current)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            # Revert
            current[idx] *= -1
        
        T *= alpha
        if T < T_end:
            break
            
    return best, best_score

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _all_reversals(coeffs):
    """Generate all reversal variants."""
    n = len(coeffs)
    variants = []
    for shift in range(n):
        rotated = coeffs[shift:] + coeffs[:shift]
        variants.append(rotated)
        variants.append(list(reversed(rotated)))
    return variants

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Generate initial candidates from known constructions
    candidates = []
    
    # Rudin-Shapiro and variants
    grs = _grs_sequence(24)
    candidates.append(grs)
    candidates.append(list(reversed(grs)))
    candidates.append([-c for c in grs])
    candidates.append([-c for c in reversed(grs)])
    
    # Try all shifts and reversals of GRS
    for variant in _all_reversals(grs):
        candidates.append(variant)
        candidates.append([-c for c in variant])
    
    # Random candidates
    random.seed(42)
    for _ in range(50):
        candidates.append([random.choice([-1, 1]) for _ in range(24)])
    
    # Additional structured candidates
    # Pattern-based: try various periodic patterns
    for period in range(2, 13):
        base = [random.choice([-1, 1]) for _ in range(period)]
        seq = [base[i % period] for i in range(24)]
        candidates.append(seq)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Phase 1: Quick screening with fast scoring
    best_coeffs = None
    best_score = float('inf')
    
    for c in unique_candidates:
        if time.time() > deadline - 60:
            break
        s = _score_fast(c, 1024)
        if s < best_score:
            best_score = s
            best_coeffs = list(c)
    
    # Phase 2: Simulated Annealing from top candidates
    top_candidates = []
    scored = [(c, _score_fast(c, 1024)) for c in unique_candidates]
    scored.sort(key=lambda x: x[1])
    
    for c, s in scored[:10]:
        if time.time() > deadline - 60:
            break
        coeffs, score = _simulated_annealing(c, time_limit=3.0)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Phase 3: Random restarts with Simulated Annealing
    if time.time() < deadline - 30:
        num_restarts = 100
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break
            coeffs = [random.choice([-1, 1]) for _ in range(24)]
            coeffs, score = _simulated_annealing(coeffs, time_limit=1.5)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # Phase 4: Iterated Local Search (Perturb + SA) from best
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            perturbed = list(best_coeffs)
            # Stronger perturbation
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, 23)
                perturbed[idx] = -perturbed[idx]
            coeffs, score = _simulated_annealing(perturbed, time_limit=1.0)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # Phase 5: Final refinement with full precision scoring
    if time.time() < deadline - 10:
        coeffs, score = _local_search(best_coeffs, time_limit=min(8.0, deadline - time.time() - 5), fast=False)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Final verification
    final_score = _score(best_coeffs)
    print(f"Final score: {final_score:.9f}")
    
    return best_coeffs
# EVOLVE-BLOCK-END
