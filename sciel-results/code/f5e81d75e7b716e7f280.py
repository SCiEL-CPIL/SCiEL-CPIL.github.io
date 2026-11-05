"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

def _square_nonzero_terms(coeffs):
    """Compute square of polynomial given as list of (exponent, coeff) pairs.
    Returns number of nonzero terms in the square."""
    # coeffs: list of (exp, coeff)
    n = len(coeffs)
    # Use a dict to accumulate
    sq = {}
    for i in range(n):
        ei, ci = coeffs[i]
        for j in range(i, n):
            ej, cj = coeffs[j]
            pos = ei + ej
            val = ci * cj
            if i != j:
                val *= 2
            sq[pos] = sq.get(pos, 0) + val
    # Count nonzero
    return sum(1 for v in sq.values() if v != 0)

def _construct_from_signs_and_exps(exps, signs):
    """Build polynomial and return (coeff_list, sq_terms)."""
    coeffs = [(exps[i], signs[i]) for i in range(len(exps)) if signs[i] != 0]
    if len(coeffs) != 12:
        return None, 999
    # Build coefficient list in ascending degree
    max_deg = max(e for e, c in coeffs)
    if max_deg >= 41:
        return None, 999
    coeff_list = [0] * (max_deg + 1)
    for e, c in coeffs:
        coeff_list[e] = c
    # Trim trailing zeros (but keep at least 1)
    while len(coeff_list) > 1 and coeff_list[-1] == 0:
        coeff_list.pop()
    # Verify: exactly 12 nonzero, last nonzero, all in [-64,64]
    nz = sum(1 for c in coeff_list if c != 0)
    if nz != 12:
        return None, 999
    if coeff_list[-1] == 0:
        return None, 999
    if any(c < -64 or c > 64 for c in coeff_list):
        return None, 999
    if len(coeff_list) > 41:
        return None, 999
    # Compute square terms
    sq_terms = _square_nonzero_terms(coeffs)
    return coeff_list, sq_terms

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_coeffs = None
    best_sq = 999
    
    # Strategy 1: Brute-force all 2^12 sign patterns for arithmetic progression exponents
    # Try different starting exponents and step sizes
    # For AP: exps = [a, a+d, a+2d, ..., a+11d]
    # We need max exp = a + 11d <= 40 (since length <= 41, indices 0..40)
    
    # First, try d=1 (consecutive), various a
    # Then try other d values
    
    # Precompute: for a given set of exponents, we can compute the square structure
    # The key is: for AP with step d, the "collision pattern" depends only on d and the sign pattern
    # Actually, for any fixed exponent set, the mapping from pairs to positions is fixed
    
    # Let's try: for each candidate exponent set, brute force 2^12 sign patterns
    # 2^12 = 4096, very fast
    
    # Candidate exponent sets to try:
    candidate_exps = []
    
    # Arithmetic progressions
    for d in range(1, 4):  # step size 1, 2, 3
        for a in range(0, 41 - 11*d):
            exps = [a + i*d for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(tuple(exps))
    
    # Some structured sets: e.g., two clusters
    # Exponents: 0,1,2,3,4,5, 20,21,22,23,24,25
    candidate_exps.append(tuple([0,1,2,3,4,5,20,21,22,23,24,25]))
    candidate_exps.append(tuple([0,1,2,3,4,5,10,11,12,13,14,15]))
    
    # Random-ish structured sets
    import random
    random.seed(42)
    for _ in range(50):
        exps = sorted(random.sample(range(0, 41), 12))
        candidate_exps.append(tuple(exps))
    
    # Deduplicate
    candidate_exps = list(set(candidate_exps))
    
    # For each candidate, brute force signs
    for exps in candidate_exps:
        if time.time() > deadline:
            break
        # Brute force 2^12 sign patterns
        for signs_tuple in product([1, -1], repeat=12):
            if time.time() > deadline:
                break
            coeffs = [(exps[i], signs_tuple[i]) for i in range(12)]
            sq_terms = _square_nonzero_terms(coeffs)
            if sq_terms < best_sq:
                best_sq = sq_terms
                coeff_list = [0] * (max(exps) + 1)
                for e, c in coeffs:
                    coeff_list[e] = c
                while len(coeff_list) > 1 and coeff_list[-1] == 0:
                    coeff_list.pop()
                if coeff_list[-1] != 0 and len(coeff_list) <= 41:
                    best_coeffs = list(coeff_list)
    
    # If we found something good, return it
    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback: return the seed
    return [1] * 12
# EVOLVE-BLOCK-END
