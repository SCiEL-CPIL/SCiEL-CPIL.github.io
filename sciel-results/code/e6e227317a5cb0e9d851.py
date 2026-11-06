import time
import random
import itertools
import numpy as np

def _square_count(coeffs_list):
    """
    Calculate the number of non-zero terms in the square of a polynomial.
    """
    if not coeffs_list:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs_list, coeffs_list)
    return int(np.count_nonzero(sq))

def _evaluate_signs(exps, signs, coeffs_mags):
    """
    Construct polynomial from exponents, signs, and magnitudes, return square count.
    """
    max_exp = max(exps)
    poly = [0] * (max_exp + 1)
    for e, s, c in zip(exps, signs, coeffs_mags):
        poly[e] = s * c
    return _square_count(poly)

def _exhaustive_sign_search(exps, coeffs_mags, deadline):
    """
    Exhaustively search all 2^12 sign patterns for a fixed exponent set and magnitudes.
    Returns best (score, signs).
    """
    best_score = 1000
    best_signs = None
    
    # 2^12 = 4096 patterns. Very fast.
    for mask in range(4096):
        if time.time() > deadline:
            break
        signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        score = _evaluate_signs(exps, signs, coeffs_mags)
        if score < best_score:
            best_score = score
            best_signs = signs
            if best_score == 12: # Theoretical lower bound for 12 terms? (12 diagonal terms min)
                break
                
    return best_score, best_signs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_poly = None
    best_score = 1000
    
    # Strategy:
    # 1. Generate diverse exponent structures.
    #    - Consecutive APs (0..11, 1..12, etc.)
    #    - Sparse APs (step 2, 3)
    #    - Random sets of 12 exponents within [0, 40]
    #    - Structured sets: e.g., two clusters, symmetric sets.
    
    exponent_candidates = []
    
    # 1. Consecutive APs
    for start in range(0, 10):
        exps = list(range(start, start + 12))
        if max(exps) <= 40:
            exponent_candidates.append(tuple(exps))
            
    # 2. Sparse APs
    for step in [2, 3, 4]:
        for start in range(0, 10):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                exponent_candidates.append(tuple(exps))
                
    # 3. Random sets
    # We need to sample sets of 12 exponents from 0..40.
    # Total combos C(41, 12) is huge. We sample.
    random.seed(42) # Deterministic for reproducibility in this block, but construct is called once?
    # Actually, we want to use time to search more if possible, but deterministic is safer for "finding" a specific good one.
    # Let's generate a few hundred random sets.
    for _ in range(500):
        exps = sorted(random.sample(range(0, 41), 12))
        exponent_candidates.append(tuple(exps))
        
    # 4. Symmetric sets (around center 20)
    # e.g., {0, 40}, {1, 39}, etc.
    for k in range(6):
        exps = []
        for i in range(k):
            exps.append(i)
            exps.append(40 - i)
        # Need 12 terms. If k=6, we have 12 terms.
        if len(exps) == 12:
            exps.sort()
            exponent_candidates.append(tuple(exps))
            
    # Remove duplicates
    exponent_candidates = list(set(exponent_candidates))
    
    # Shuffle to ensure we don't bias towards the first ones if we run out of time
    random.shuffle(exponent_candidates)
    
    # Coefficient magnitudes to try:
    # Mostly 1s, but some 2s might help? 
    # The task allows [-64, 64]. 
    # However, for sparse squares, coefficients of 1 are usually best for cancellation.
    # Let's primarily test all 1s.
    # If we have time, we can test some 2s.
    
    coeffs_mags_candidates = [
        [1]*12,
        [2]*12,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], # Just in case
    ]
    
    # Search
    for exps in exponent_candidates:
        if time.time() > deadline - 2.0:
            break
            
        for coeffs in coeffs_mags_candidates:
            score, signs = _exhaustive_sign_search(exps, coeffs, deadline)
            
            if score < best_score:
                best_score = score
                # Reconstruct poly
                max_exp = max(exps)
                poly = [0] * (max_exp + 1)
                for e, s, c in zip(exps, signs, coeffs):
                    poly[e] = s * c
                # Ensure length <= 41 and final coeff non-zero
                if len(poly) <= 41 and poly[-1] != 0:
                    best_poly = poly
                    
                if best_score <= 13:
                    # Very good. 12 is theoretical min? 
                    # Square of 12 terms has 12 diagonal terms. 
                    # If all off-diagonals cancel, we have 12.
                    # If 12 is found, break.
                    if best_score == 12:
                        return best_poly
                    # Continue searching for 12 if possible? 
                    # Let's break if 12 is found.
                    if best_score == 12:
                        break

    # If we haven't found a good one, do a quick random local search on the best found so far
    if best_poly is None:
        # Fallback: simple AP
        poly = [0]*12
        for i in range(12):
            poly[i] = 1 if i % 2 == 0 else -1
        return poly

    # Refine best_poly with a few random sign flips if time permits
    if time.time() < deadline - 5.0:
        current_poly = best_poly[:]
        current_score = best_score
        nz_indices = [i for i, v in enumerate(current_poly) if v != 0]
        
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            new_val = -old_val # Flip sign
            current_poly[idx] = new_val
            new_score = _square_count(current_poly)
            if new_score < current_score:
                current_score = new_score
                best_poly = current_poly[:]
                best_score = current_score
            else:
                current_poly[idx] = old_val

    return best_poly
