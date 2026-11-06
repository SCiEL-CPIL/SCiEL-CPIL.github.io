import time
import random
import itertools

def is_sidon(subset):
    """Check if subset has all unordered pair sums (i<=j) unique."""
    s = sorted(subset)
    n = len(s)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen_sums:
                return False
            seen_sums.add(sm)
    return True

def max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # We want to minimize the max Sidon subset size.
    # Small ranges force collisions, potentially reducing the max Sidon size.
    # We search exhaustively in small ranges.
    
    best_val = 12
    best_set = list(range(12))
    
    # Ranges to search: [0, 11], [0, 12], [0, 13], [0, 14], [0, 15]
    # Number of combinations:
    # [0, 11]: C(12, 12) = 1
    # [0, 12]: C(13, 12) = 13
    # [0, 13]: C(14, 12) = 91
    # [0, 14]: C(15, 12) = 455
    # [0, 15]: C(16, 12) = 1820
    
    for max_val in [11, 12, 13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Generate all 12-subsets of [0, max_val]
        for comb in itertools.combinations(range(max_val + 1), 12):
            if time.time() > deadline:
                break
                
            val = max_sidon_size(comb)
            if val < best_val:
                best_val = val
                best_set = list(comb)
                
                # If we found a set with max Sidon size 5, 
                # it's very hard to go lower (4 requires avoiding all 5-subsets being Sidon).
                # Given the constraints, 5 is a strong candidate for improvement over 6.
                if best_val <= 5:
                    return best_set

    return best_set
