import numpy as np
import random

def _square_nonzero_count(coeffs):
    """Compute number of nonzero terms in Q^2 given dense coefficient list of Q."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_coeffs = None
    
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

    random.seed(42)
    
    # 1. Try specific patterns first
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
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
    ]
    for pat in patterns:
        if any(c == 0 for c in pat):
            continue
        try_candidate(pat)

    # 2. Random restarts with stronger local search
    for restart in range(300):
        # Initialize with random small integers
        c = [random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) for _ in range(12)]
        try_candidate(c)
        
        improved = True
        # Local search: try changing each coefficient to a value in a range
        while improved:
            improved = False
            for i in range(12):
                # Try a range of values around current value
                # And some global values
                candidates = set()
                for delta in range(-5, 6):
                    candidates.add(c[i] + delta)
                candidates.add(-c[i])
                candidates.add(c[i] * 2 if abs(c[i] * 2) <= 64 else c[i])
                candidates.add(c[i] // 2 if c[i] % 2 == 0 else c[i])
                
                # Add some random values
                for _ in range(5):
                    candidates.add(random.randint(-20, 20))
                
                for new_c in candidates:
                    if new_c == 0:
                        continue
                    if abs(new_c) > 64:
                        continue
                    if new_c == c[i]:
                        continue
                    new_c_list = list(c)
                    new_c_list[i] = new_c
                    new_count = _square_nonzero_count(new_c_list)
                    if new_count < best_count:
                        try_candidate(new_c_list)
                    if new_count < _square_nonzero_count(c):
                        c = new_c_list
                        improved = True
                        break

    # Build result
    # Use exponents 0, 1, ..., 11.
    result = [0] * 12
    for i, c in enumerate(best_coeffs):
        result[i] = c
        
    # Ensure last is non-zero
    if result[-1] == 0:
        result[-1] = 1
        
    return result
