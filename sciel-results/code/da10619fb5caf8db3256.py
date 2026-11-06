"""
Strategy: Combinatorial optimization over exponent multisets.
1. Enumerate candidate exponent sets (12 terms).
2. Filter by "sumset size" (number of distinct sums e_i + e_j). 
   A smaller sumset means fewer potential nonzero terms in the square.
3. For the most promising exponent sets, perform a local search on 
   coefficients (small integers) to minimize the actual nonzero count 
   in the square, exploiting cancellations.
"""
import itertools
import random
import sys
import time
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 82.0

def _is_deadline():
    return time.time() >= _DEADLINE

def _square_count(coeffs):
    """Count nonzero terms in the square of a polynomial given as a list of coefficients."""
    # Use numpy for speed if possible, but manual is fine for small sizes
    # coeffs is a list where index is degree.
    n = len(coeffs)
    if n == 0:
        return 0
    # Convolution
    # We only care about the count of non-zeros.
    # Using numpy conv is faster.
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _normalize(poly):
    """Validate and normalize a polynomial list."""
    if not poly:
        return None
    # Trim trailing zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    if len(poly) > 41:
        return None
    if poly[-1] == 0:
        return None
    # Exactly 12 nonzero terms
    nz = sum(1 for c in poly if c != 0)
    if nz != 12:
        return None
    # Entries in [-64, 64]
    for c in poly:
        if not (-64 <= c <= 64):
            return None
    return poly

def _sumset_size(exponents):
    """Calculate the number of distinct sums e_i + e_j for i <= j."""
    sums = set()
    n = len(exponents)
    for i in range(n):
        for j in range(i, n):
            sums.add(exponents[i] + exponents[j])
    return len(sums)

def _generate_exponent_candidates():
    """
    Generate candidate exponent sets of size 12 within [0, 40].
    We prioritize sets that look like arithmetic progressions or have small sumsets.
    """
    candidates = []
    
    # 1. Arithmetic Progressions (APs)
    # P(x) = sum a_i x^{a + i*d}
    # If we choose coefficients carefully, cancellations might happen.
    # Sumset of an AP {a, a+d, ..., a+11d} is {2a, 2a+d, ..., 2a+22d}. Size = 23.
    for a in range(0, 41):
        for d in range(1, 10):
            exps = [a + i * d for i in range(12)]
            if exps[-1] <= 40:
                candidates.append(tuple(exps))
                
    # 2. Random sets, filtered by sumset size
    rng = random.Random(42)
    seen = set()
    for _ in range(20000):
        if _is_deadline():
            break
        # Sample 12 distinct exponents from 0..40
        exps = tuple(sorted(rng.sample(range(0, 41), 12)))
        if exps in seen:
            continue
        seen.add(exps)
        
        # Heuristic: sumset size should be small. 
        # Minimum possible sumset size for 12 distinct integers is 23 (AP).
        # So we look for sumset size <= 30 or so.
        ss = _sumset_size(exps)
        if ss <= 35:
            candidates.append(exps)
            
    # 3. Structured "clumps"
    # Two blocks of 6 terms far apart?
    # Sumset would be union of sumsets of blocks plus cross terms.
    # If blocks are far, cross terms are distinct.
    # This might not be optimal for sparsity, but worth checking.
    # Actually, APs are best for minimizing sumset size.
    
    # Deduplicate
    unique_candidates = list(set(candidates))
    return unique_candidates

def _local_search_coefficients(exponents, initial_coeffs, max_steps=100):
    """
    Given fixed exponents, search for coefficients that minimize square nonzero count.
    Uses a simple hill-climbing / random restart approach on a small subset of coefficients.
    """
    n_terms = len(exponents)
    max_degree = max(exponents)
    
    # Initialize polynomial
    poly = [0] * (max_degree + 1)
    for i, exp in enumerate(exponents):
        poly[exp] = initial_coeffs[i]
        
    current_score = _square_count(poly)
    best_score = current_score
    best_poly = list(poly)
    
    if best_score <= 2:
        return best_poly, best_score

    rng = random.Random(hash(tuple(exponents)) % 10000)
    
    # Coefficients are integers in [-64, 64].
    # Small integers are more likely to cause cancellations (e.g., 1, -1, 2, -2).
    # We will perturb one coefficient at a time.
    
    steps = 0
    while steps < max_steps and not _is_deadline():
        steps += 1
        # Pick a random term to change
        idx = rng.randint(0, n_terms - 1)
        old_val = poly[exponents[idx]]
        
        # Try changing to a nearby value or a random small value
        # Strategy: try +/- 1, or random from {-3,-2,-1,1,2,3}
        if rng.random() < 0.5:
            new_val = old_val + rng.choice([-1, 1])
        else:
            new_val = rng.choice([-3, -2, -1, 1, 2, 3])
            
        if new_val == old_val:
            continue
        if not (-64 <= new_val <= 64):
            continue
            
        poly[exponents[idx]] = new_val
        new_score = _square_count(poly)
        
        if new_score < best_score:
            best_score = new_score
            best_poly = list(poly)
            if best_score <= 2:
                break
        else:
            # Revert if not better (greedy)
            # Or keep with some probability (simulated annealing style)
            # For simplicity, greedy with occasional random jump
            if rng.random() < 0.1:
                # Keep the change (random walk)
                pass
            else:
                poly[exponents[idx]] = old_val
                
    return best_poly, best_score

def _search():
    best_poly = None
    best_score = 10000
    
    candidates = _generate_exponent_candidates()
    
    # Sort candidates by sumset size (lower is better)
    # We don't want to compute sumset for all if it's expensive, but 12 terms is small.
    scored_candidates = []
    for c in candidates:
        if _is_deadline():
            break
        ss = _sumset_size(c)
        scored_candidates.append((ss, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # We can't test all. Prioritize the smallest sumsets.
    # Limit the number of exponent sets we deeply search.
    max_sets_to_search = 500
    
    for ss, exps in scored_candidates[:max_sets_to_search]:
        if _is_deadline():
            break
            
        # Initial coefficients: all 1s
        init_coeffs = [1] * 12
        
        # Run local search
        poly, score = _local_search_coefficients(exps, init_coeffs, max_steps=50)
        
        norm = _normalize(poly)
        if norm is not None:
            if score < best_score:
                best_score = score
                best_poly = norm
                if best_score <= 2:
                    return best_poly
                    
        # Try a few different initializations for the same exponents
        # e.g., alternating signs
        init_coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
        poly_alt, score_alt = _local_search_coefficients(exps, init_coeffs_alt, max_steps=20)
        norm_alt = _normalize(poly_alt)
        if norm_alt is not None:
            if score_alt < best_score:
                best_score = score_alt
                best_poly = norm_alt
                if best_score <= 2:
                    return best_poly
                    
    if best_poly is not None:
        return best_poly
        
    # Fallback: A valid 12-term polynomial
    # Use an AP with all 1s
    fallback_exps = [i for i in range(12)]
    fallback = [0] * 12
    for i in range(12):
        fallback[i] = 1
    return fallback

def construct():
    """Return coefficient list in ascending degree, length<=41, nonzero final
    coefficient, entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square."""
    result = _search()
    if result is None:
        result = [1] * 12
    return result
