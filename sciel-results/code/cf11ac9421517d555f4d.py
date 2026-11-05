import numpy as np
import random

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in Q^2 given dense coefficient list of Q."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_coeffs = None
    
    # We assume an Arithmetic Progression for exponents to minimize the support size of the square.
    # The number of non-zero terms in P(x)^2 is equal to the number of non-zero terms in Q(y)^2
    # where Q(y) has the same coefficients as P(x) but with consecutive exponents 0..11.
    # We search for the best 12 coefficients for Q(y).
    
    def try_candidate(coeffs):
        nonlocal best_count, best_coeffs
        if any(c == 0 for c in coeffs):
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        count = _square_nonzero_count(coeffs)
        if count < best_count:
            best_count = count
            best_coeffs = list(coeffs)

    # Random search with local improvement
    random.seed(42)
    
    for restart in range(200):
        # Initialize with random small integers
        c = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        try_candidate(c)
        
        # Local search
        improved = True
        while improved:
            improved = False
            for i in range(12):
                # Try perturbing the coefficient
                candidates = [c[i] - 1, c[i] + 1, -c[i]]
                for _ in range(3):
                    candidates.append(random.randint(-10, 10))
                
                for new_c in candidates:
                    if new_c == 0:
                        continue
                    if abs(new_c) > 64:
                        continue
                    if new_c == c[i]:
                        continue
                    new_c_list = list(c)
                    new_c_list[i] = new_c
                    old_count = _square_nonzero_count(c)
                    new_count = _square_nonzero_count(new_c_list)
                    if new_count < old_count:
                        c = new_c_list
                        improved = True
                        break
                    elif new_count < best_count:
                        try_candidate(new_c_list)
                        if new_count < old_count:
                             c = new_c_list
                             improved = True
                             break

    # Also try specific patterns
    patterns = [
        [1]*12,
        [-1]*12,
        [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [1,2,-2,-1,-1,2,1,-2,2,-1,1,-1],
        [1,-2,3,-4,5,-6,7,-8,9,-10,11,-12],
        [1,1,1,1,1,1,1,1,1,1,-1,-1],
        [1,1,1,1,-1,-1,-1,-1,1,1,1,1],
    ]
    for pat in patterns:
        if any(c == 0 for c in pat):
            continue
        try_candidate(pat)

    # Build result
    # Use exponents 0, 1, ..., 11. Max degree 11.
    # The result is the coefficient list of P(x).
    result = [0] * 12
    for i, c in enumerate(best_coeffs):
        result[i] = c
        
    # Ensure last is non-zero
    if result[-1] == 0:
        result[-1] = 1
        
    return result
