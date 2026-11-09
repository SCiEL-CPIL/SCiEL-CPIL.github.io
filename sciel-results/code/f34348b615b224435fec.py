import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
    
    # Self-inverse elements: 1 and 1008
    # We will search starts from 2 to 1007 to avoid self-inverses
    
    # Build pairs_by_a
    # pairs_by_a[a] contains b such that (a, b) is a conflict pair with a < b
    pairs_by_a = [[] for _ in range(LIMIT + 1)]
    
    for i in range(2, 1008): # 2 to 1007
        j = inv[i]
        if j == i:
            continue
        a, b = min(i, j), max(i, j)
        # a is the smaller one, b is the larger one
        # We add b to pairs_by_a[a]
        pairs_by_a[a].append(b)
        
    # Compute min_b_limit[s]
    # min_b_limit[s] = min { b | (a, b) in pairs, a >= s }
    # If no such pair, min_b_limit[s] = 1008 (infinity)
    min_b_limit = [1008] * (LIMIT + 1)
    
    current_min_b = 1008
    # Iterate s from 1008 down to 1
    # When we are at s, we include all pairs with a == s
    for s in range(1008, 0, -1):
        for b in pairs_by_a[s]:
            if b < current_min_b:
                current_min_b = b
        min_b_limit[s] = current_min_b
        
    best_len = 0
    best_start = 2
    
    # Search for best start
    # s must be such that s is not self-inverse.
    # In range [1, 1008], only 1 and 1008 are self-inverse.
    # So s in [2, 1007] is safe.
    # Also we need the interval to be contained in [1, 1008].
    # max_e = min_b_limit[s] - 1
    # If min_b_limit[s] is 1008, max_e = 1007.
    # If min_b_limit[s] is smaller, max_e is smaller.
    # We require max_e <= 1008, which is always true.
    # We require max_e >= s, which is always true as argued.
    
    for s in range(2, 1008):
        # Check time
        if time.time() - start_time > 80:
            break
            
        max_e = min_b_limit[s] - 1
        # Ensure max_e does not exceed 1008 (it won't, min_b_limit is at least s+1 or 1008)
        # But just in case, clamp to 1008
        if max_e > 1008:
            max_e = 1008
            
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
