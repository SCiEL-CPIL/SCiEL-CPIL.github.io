"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _square_poly(p):
    """Compute exact integer square of polynomial given as coefficient list (ascending degree)."""
    n = len(p)
    # Result has degree up to 2*(n-1), so length 2*n-1
    result = [0] * (2 * n - 1)
    for i in range(n):
        if p[i] == 0:
            continue
        for j in range(n):
            if p[j] == 0:
                continue
            result[i + j] += p[i] * p[j]
    return result

def _trim(coeffs):
    """Remove trailing zeros, keep at least 1 element."""
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return coeffs

def _is_valid_candidate(coeffs):
    """Check: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if not coeffs or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    if _count_nonzero(coeffs) != 12:
        return False
    return True

def _score_candidate(coeffs):
    """Return number of nonzero terms in the square (lower is better)."""
    sq = _square_poly(coeffs)
    sq = _trim(sq)
    return _count_nonzero(sq)

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    # Start with seed
    best_coeffs = [1] * 12
    best_nz = _score_candidate(best_coeffs)  # Should be 23
    
    # Try some structured candidates first
    structured = []
    
    # Alternating signs
    for k in range(12):
        p = [1 if i % 2 == 0 else -1 for i in range(k)] + [0] * (12 - k)
        # Pad to ensure 12 nonzero terms
        # Actually let's just generate 12-term polynomials
    
    # Try: 1 + x + ... + x^(a-1) - x^a - ... - x^(a+b-1) patterns
    for a in range(1, 12):
        for b in range(1, 12 - a + 1):
            total = a + b
            if total < 12:
                continue
            # Make a 12-term polynomial: first a terms +1, next (12-a) terms -1
            p = [1] * a + [-1] * (12 - a)
            # This is a contiguous 12-term polynomial
            if _is_valid_candidate(p):
                nz = _score_candidate(p)
                if nz < best_nz:
                    best_nz = nz
                    best_coeffs = p[:]
    
    # Try symmetric patterns: 1,0,...,0,1,0,...,0,1 (sparse with gaps)
    # 12 ones at positions that might create structure
    # Try: all coefficients +1 or -1 in various patterns
    
    # Random search with small coefficients
    rng = random.Random(42)
    
    # Try coefficients in {-1, 1} with various degree placements
    for attempt in range(500000):
        if time.time() > deadline:
            break
        
        # Choose 12 distinct degrees from 0 to 40
        degrees = sorted(rng.sample(range(41), 12))
        # Choose signs
        coeffs = [0] * 41
        for d in degrees:
            coeffs[d] = rng.choice([-1, 1])
        
        # Trim
        coeffs_trimmed = _trim(coeffs)
        if not _is_valid_candidate(coeffs_trimmed):
            continue
        
        nz = _score_candidate(coeffs_trimmed)
        if nz < best_nz:
            best_nz = nz
            best_coeffs = coeffs_trimmed[:]
            if best_nz <= 12:
                break
    
    # Try coefficients in {-2, -1, 1, 2}
    for attempt in range(300000):
        if time.time() > deadline:
            break
        
        degrees = sorted(rng.sample(range(41), 12))
        coeffs = [0] * 41
        for d in degrees:
            coeffs[d] = rng.choice([-2, -1, 1, 2])
        
        coeffs_trimmed = _trim(coeffs)
        if not _is_valid_candidate(coeffs_trimmed):
            continue
        
        nz = _score_candidate(coeffs_trimmed)
        if nz < best_nz:
            best_nz = nz
            best_coeffs = coeffs_trimmed[:]
            if best_nz <= 12:
                break
    
    # Try some more structured: geometric-like with gaps
    # P = 1 + x^a + x^(2a) + ... pattern won't work for 12 terms easily
    # Try: pairs of terms that cancel in square
    # If P has terms c_i*x^i and c_j*x^j, the cross term is 2*c_i*c_j*x^(i+j)
    # For cancellation, we need 2*c_i*c_j + 2*c_k*c_l = 0 for same degree i+j = k+l
    
    # Try: P = sum of 6 pairs where each pair's cross terms can cancel
    # This is complex; let's try a specific known good structure
    
    # Try: coefficients [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] type patterns
    # i.e., 12 ones at even positions
    for gap in [2, 3, 4, 5, 6, 7, 8]:
        if 11 * gap >= 41:
            continue
        coeffs = [0] * (11 * gap + 1)
        for i in range(12):
            coeffs[i * gap] = 1
        coeffs_trimmed = _trim(coeffs)
        if _is_valid_candidate(coeffs_trimmed):
            nz = _score_candidate(coeffs_trimmed)
            if nz < best_nz:
                best_nz = nz
                best_coeffs = coeffs_trimmed[:]
    
    # Try mixed signs with gaps
    for gap in [2, 3, 4]:
        if 11 * gap >= 41:
            continue
        for sign_pattern in [[1]*6 + [-1]*6, [1, -1]*6, [1]*12]:
            coeffs = [0] * (11 * gap + 1)
            for i in range(12):
                coeffs[i * gap] = sign_pattern[i]
            coeffs_trimmed = _trim(coeffs)
            if _is_valid_candidate(coeffs_trimmed):
                nz = _score_candidate(coeffs_trimmed)
                if nz < best_nz:
                    best_nz = nz
                    best_coeffs = coeffs_trimmed[:]
    
    return best_coeffs
# EVOLVE-BLOCK-END
