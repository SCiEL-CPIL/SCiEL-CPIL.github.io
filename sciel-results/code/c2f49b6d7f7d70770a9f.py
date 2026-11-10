"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in the square of polynomial given by coeffs."""
    n = len(coeffs)
    # Square: result[k] = sum_{i+j=k} coeffs[i]*coeffs[j]
    # Max degree is 2*(n-1)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(i, n):
            cj = coeffs[j]
            if cj == 0:
                continue
            k = i + j
            if i == j:
                sq[k] += ci * cj
            else:
                sq[k] += 2 * ci * cj
    # Count nonzero
    return sum(1 for x in sq if x != 0)

def _is_valid_candidate(coeffs):
    """Check if coeffs is a valid candidate."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best_count = 23  # baseline: (1+x+...+x^11)^2 has 23 nonzero terms
    best_coeffs = [1]*12
    
    # Try some structured candidates
    structured = []
    
    # Pattern: 1,1,1,...,1,-1,-1,...,-1 (split signs)
    for split in range(1, 12):
        c = [1]*split + [-1]*(12-split)
        structured.append(c)
    
    # Alternating signs
    structured.append([1 if i%2==0 else -1 for i in range(12)])
    
    # Try with gaps
    for gap in range(1, 5):
        coeffs = [0]*41
        for i in range(12):
            pos = i * gap
            if pos < 41:
                coeffs[pos] = 1
        if _is_valid_candidate(coeffs):
            structured.append(coeffs)
    
    # Try P(x) = (1+x)^11 type but with coefficient variations
    # (1+x)^11 = sum C(11,k) x^k, 12 terms
    import math
    coeffs = [math.comb(11, k) for k in range(12)]
    if all(-64 <= c <= 64 for c in coeffs):
        structured.append(coeffs)
    
    # Evaluate structured candidates
    for c in structured:
        if not _is_valid_candidate(c):
            continue
        cnt = _square_nonzero_count(c)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = list(c)
        if time.time() > deadline:
            return best_coeffs
    
    # Random search
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random 12-term polynomial
        # Choose 12 distinct positions from 0..40
        positions = random.sample(range(41), 12)
        random.shuffle(positions)
        coeffs = [0]*41
        for pos in positions:
            # Random coefficient in [-64,64], nonzero
            c = random.randint(-64, 64)
            while c == 0:
                c = random.randint(-64, 64)
            coeffs[pos] = c
        if not _is_valid_candidate(coeffs):
            continue
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = list(coeffs)
            # If we find something very good, we can stop early
            if best_count <= 5:
                break
    
    # Trim trailing zeros
    while len(best_coeffs) > 1 and best_coeffs[-1] == 0:
        best_coeffs.pop()
    
    return best_coeffs
# EVOLVE-BLOCK-END
