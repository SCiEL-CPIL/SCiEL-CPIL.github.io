import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Primes p ≡ 3 (mod 4) smaller than 67
    # 31, 43, 47, 59
    # We check them in increasing order of n (decreasing score? No, smaller n is better score)
    # We want the smallest n.
    candidates = [31, 43, 47, 59]
    
    def is_valid_paley(p):
        # Compute squares
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Build out_masks
        # out_mask[v] is a bitmask where bit j is 1 if v -> j
        out_masks = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if v == u:
                    continue
                diff = (u - v) % p
                if diff in squares:
                    mask |= (1 << u)
            out_masks[v] = mask
            
        # Check all 4-sets
        # To speed up, we can iterate over combinations
        from itertools import combinations
        for S in combinations(range(p), 4):
            s_mask = 0
            for v in S:
                s_mask |= (1 << v)
            
            # Check if any v dominates S
            # v must be in V \ S, but as discussed, if v in S, bit v is not in out_mask[v], so match fails.
            found = False
            for v in range(p):
                if (out_masks[v] & s_mask) == s_mask:
                    found = True
                    break
            if not found:
                return False
        return True

    best_n = 67
    best_orientation = None
    
    # Check candidates
    for p in candidates:
        if time.time() > deadline:
            break
        if is_valid_paley(p):
            best_n = p
            # Build orientation for p
            squares = set()
            for x in range(1, p):
                squares.add((x * x) % p)
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    diff = (j - i) % p
                    if diff in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            best_orientation = orientation
            break # Found smallest valid p, stop
    
    if best_orientation is None:
        # Fall back to 67
        p = 67
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        best_orientation = orientation
        best_n = 67

    return {'n': best_n, 'orientation': best_orientation}
