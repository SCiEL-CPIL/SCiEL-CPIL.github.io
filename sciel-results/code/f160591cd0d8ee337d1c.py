"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    n = len(c)
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    return sum(1 for x in c if x != 0)

def _is_valid_poly(c):
    """Check if polynomial is valid: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(c) == 0 or len(c) > 41:
        return False
    if c[-1] == 0:
        return False
    if any(abs(x) > 64 for x in c):
        return False
    if sum(1 for x in c if x != 0) != 12:
        return False
    return True

def _basis_polynomials():
    """
    Generate basis polynomials that individually have sparse squares.
    Key insight: polynomials like (1+x+x^2+...+x^(k-1)) have squares with at most 2k-1 terms.
    Also, polynomials with symmetric supports can have cancellation in their squares.
    
    We'll use:
    - Consecutive monomial blocks: 1+x+...+x^(k-1)
    - Gapped blocks with specific patterns
    - Palindromic blocks
    """
    bases = []
    
    # Consecutive blocks of various sizes
    for start in range(0, 30):
        for size in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if start + size > 41:
                continue
            p = [0] * (start + size)
            for i in range(size):
                p[start + i] = 1
            bases.append(p)
    
    # Palindromic consecutive blocks
    for center in range(0, 20):
        for half in [2, 3, 4, 5, 6]:
            p = [0] * (2 * half + 1)
            for i in range(half + 1):
                p[i] = 1
                p[2 * half - i] = 1
            # Shift to different positions
            for shift in range(0, 20):
                shifted = [0] * shift + p
                if len(shifted) <= 41:
                    bases.append(shifted)
    
    # Gapped patterns: every other term
    for start in range(0, 25):
        for size in [4, 5, 6, 7, 8]:
            p = [0] * (2 * size)
            for i in range(size):
                p[start + 2 * i] = 1
            if len(p) <= 41:
                bases.append(p)
    
    # Specific known good patterns
    special = [
        # (1+x)(1+x^2)(1+x^4) = 1+x+x^2+x^3+x^4+x^5+x^6+x^7
        [1,1,1,1,1,1,1,1],
        # (1+x+x^2)(1+x^3+x^6)
        [1,1,1,1,0,0,1,0,0,1],
        # Fibonacci-like
        [1,1,0,1,0,1,0,0,1,0,0,0,1],
    ]
    for s in special:
        if len(s) <= 41:
            bases.append(s)
    
    return bases

def _combine_bases(bases, weights):
    """
    Combine basis polynomials with given weights.
    Each weight is a list of (basis_index, coefficient) pairs.
    Returns the combined polynomial.
    """
    result = [0] * 41
    for bi, w in weights:
        b = bases[bi]
        for i, c in enumerate(b):
            if c != 0:
                result[i] += w * c
    return result

def _search_weight_combinations(bases, deadline, best_sq_nnz, best_poly):
    """
    Search over combinations of 2-3 basis polynomials with small integer weights.
    The key is that cross-terms between different bases may cancel.
    """
    n_bases = len(bases)
    rng = random.Random(999)
    
    # Pre-filter: only consider bases that don't overlap too much
    # For pairs, try all combinations of non-identical bases
    # For triples, sample
    
    # Phase A: Try pairs of bases with weights in {-3,-2,-1,1,2,3}
    weight_vals = [-3, -2, -1, 1, 2, 3]
    
    # Sample pairs to avoid O(n^2) explosion
    pair_samples = 50000
    attempts = 0
    
    while time.time() < deadline and attempts < pair_samples:
        attempts += 1
        if attempts % 10000 == 0:
            if time.time() > deadline:
                break
        
        # Pick two different bases
        i1 = rng.randint(0, n_bases - 1)
        i2 = rng.randint(0, n_bases - 1)
        if i1 == i2:
            continue
        
        w1 = rng.choice(weight_vals)
        w2 = rng.choice(weight_vals)
        
        poly = _combine_bases(bases, [(i1, w1), (i2, w2)])
        
        # Trim trailing zeros
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        
        if not _is_valid_poly(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # Phase B: Try triples of bases
    triple_samples = 30000
    attempts = 0
    
    while time.time() < deadline and attempts < triple_samples:
        attempts += 1
        if attempts % 5000 == 0:
            if time.time() > deadline:
                break
        
        # Pick three different bases
        idxs = rng.sample(range(n_bases), min(3, n_bases))
        if len(idxs) < 3:
            continue
        
        w1 = rng.choice(weight_vals)
        w2 = rng.choice(weight_vals)
        w3 = rng.choice(weight_vals)
        
        poly = _combine_bases(bases, [(idxs[0], w1), (idxs[1], w2), (idxs[2], w3)])
        
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        
        if not _is_valid_poly(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    return best_sq_nnz, best_poly

def _local_search(poly, deadline, best_sq_nnz, best_poly):
    """
    Local search: perturb coefficients by small amounts, check validity, evaluate square nnz.
    Returns updated (best_sq_nnz, best_poly).
    """
    c = poly[:]
    current_sq_nnz = best_sq_nnz
    
    # Precompute square for current
    sq = _square_coeffs(c)
    cur_nnz = _nnz(sq)
    if cur_nnz < current_sq_nnz:
        current_sq_nnz = cur_nnz
        best_poly = c[:]
    
    # Local search: try flipping signs, incrementing/decrementing by 1,2,3
    steps = [1, -1, 2, -2, 3, -3]
    improved = True
    max_rounds = 100
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try each position
        for i in range(len(c)):
            if time.time() > deadline:
                break
            
            old_val = c[i]
            for step in steps:
                new_val = old_val + step
                if abs(new_val) > 64:
                    continue
                if new_val == old_val:
                    continue
                
                c[i] = new_val
                # Check validity quickly
                if not _is_valid_poly(c):
                    c[i] = old_val
                    continue
                
                sq = _square_coeffs(c)
                nnz_sq = _nnz(sq)
                
                if nnz_sq < current_sq_nnz:
                    current_sq_nnz = nnz_sq
                    best_poly = c[:]
                    improved = True
                else:
                    c[i] = old_val
    
    return current_sq_nnz, best_poly

def _structured_candidates():
    """Generate structured candidate polynomials likely to have sparse squares."""
    candidates = []
    
    # 1. All ones degree 11 (palindromic)
    candidates.append([1]*12)
    
    # 2. Alternating signs
    candidates.append([1,-1,1,-1,1,-1,-1,1,-1,1,-1,1])
    
    # 3. Palindromic patterns with small coefficients
    patterns = [
        [1,1,1,1,1,1],
        [1,1,1,-1,-1,-1],
        [1,-1,1,-1,1,-1],
        [1,2,1,2,1,2],
        [2,1,2,1,2,1],
        [1,1,-1,1,1,-1],
        [1,-1,-1,-1,-1,1],
        [3,1,1,1,1,3],
        [1,3,1,1,3,1],
        [1,1,3,1,1,3],
        [1,2,3,2,1,2],
        [2,1,2,3,1,2],
        [1,1,1,2,1,1],
        [1,2,1,1,2,1],
        [1,1,2,1,1,2],
    ]
    for p in patterns:
        pal = p + [p[4], p[3], p[2], p[1], p[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    # 4. Try polynomials of the form sum of specific monomials
    supports = [
        [13,12,11,10,9,8,7,6,5,4,2,0],
        [13,12,11,10,9,8,7,6,5,3,1,0],
        [13,12,11,10,9,8,7,6,4,2,1,0],
        [13,12,11,10,9,8,7,5,4,3,2,1],
        [13,12,11,10,9,8,6,5,4,3,2,1],
        [13,12,11,10,9,7,6,5,4,3,2,1],
        [13,12,11,10,8,7,6,5,4,3,2,1],
        [13,12,11,9,8,7,6,5,4,3,2,1],
        [13,12,10,9,8,7,6,5,4,3,2,1],
        [13,11,10,9,8,7,6,5,4,3,2,1],
    ]
    for sup in supports:
        poly = [0]*14
        for d in sup:
            poly[d] = 1
        if _is_valid_poly(poly):
            candidates.append(poly)
    
    # 5. Try with coefficient -1 for some terms
    for sup in supports[:5]:
        poly = [0]*14
        for j, d in enumerate(sup):
            poly[d] = 1 if j % 2 == 0 else -1
        if _is_valid_poly(poly):
            candidates.append(poly)
    
    # 6. Degree 15 palindromic with 12 nonzero terms
    pal15_patterns = [
        [1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1],
        [1,1,0,0,1,1,0,0],
        [1,0,0,1,0,0,1,0],
        [1,1,1,0,1,1,1,0],
    ]
    for p in pal15_patterns:
        pal = p + [p[6], p[5], p[4], p[3], p[2], p[1], p[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    # 7. Random palindromic with coefficients from {-2,-1,1,2}
    rng = random.Random(12345)
    for _ in range(200):
        free = [rng.choice([-2,-1,1,2]) for _ in range(6)]
        pal = free + [free[4], free[3], free[2], free[1], free[0]]
        if _is_valid_poly(pal):
            candidates.append(pal)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    best_sq_nnz = float('inf')
    best_poly = [1]*12
    
    # Phase 1: Evaluate structured candidates
    candidates = _structured_candidates()
    for c in candidates:
        if time.time() > deadline:
            break
        if not _is_valid_poly(c):
            continue
        sq = _square_coeffs(c)
        nnz_sq = _nnz(sq)
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = c[:]
    
    # Phase 2: NEW - Basis combination search
    if time.time() < deadline:
        bases = _basis_polynomials()
        best_sq_nnz, best_poly = _search_weight_combinations(bases, deadline, best_sq_nnz, best_poly)
    
    # Phase 3: Random search on palindromic degree-11 polynomials
    non_zero_vals = list(range(-8, 9))
    non_zero_vals.remove(0)
    
    rng = random.Random(42)
    attempts = 0
    max_attempts = 300000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 50000 == 0 and time.time() > deadline:
            break
        
        # Pick 6 nonzero coefficients
        free = [rng.choice(non_zero_vals) for _ in range(6)]
        # Build palindromic degree 11: c0..c5 = free, c6..c11 = free[5],free[4],...,free[0]
        poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        
        if not _is_valid_poly(poly):
            continue
        
        sq = _square_coeffs(poly)
        nnz_sq = _nnz(sq)
        
        if nnz_sq < best_sq_nnz:
            best_sq_nnz = nnz_sq
            best_poly = poly[:]
    
    # Phase 4: Local search from best found
    if time.time() < deadline:
        best_sq_nnz, best_poly = _local_search(best_poly, deadline, best_sq_nnz, best_poly)
    
    # Phase 5: Additional random search with different structure
    if time.time() < deadline:
        for _ in range(10000):
            if time.time() > deadline:
                break
            degrees = rng.sample(range(0, 41), 12)
            degrees.sort()
            if degrees[-1] == 0:
                continue
            coeffs = [rng.choice([-2,-1,1,2,3]) for _ in range(12)]
            poly = [0]*41
            for d, c in zip(degrees, coeffs):
                poly[d] = c
            # Trim trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            if not _is_valid_poly(poly):
                continue
            sq = _square_coeffs(poly)
            nnz_sq = _nnz(sq)
            if nnz_sq < best_sq_nnz:
                best_sq_nnz = nnz_sq
                best_poly = poly[:]
    
    # Final local search
    if time.time() < deadline:
        best_sq_nnz, best_poly = _local_search(best_poly, deadline, best_sq_nnz, best_poly)
    
    # Ensure output is valid
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if len(best_poly) == 0 or best_poly[-1] == 0:
        best_poly = [1]*12
    
    if sum(1 for x in best_poly if x != 0) != 12:
        best_poly = [1]*12
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    return best_poly

# EVOLVE-BLOCK-END
