import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # For each residue, find the smallest subset of {1..10} whose inverses sum to that residue
    best = [None] * 101
    min_size = [11] * 101  # 11 is infinity since max size is 10
    
    # Iterate over all non-empty subsets
    for mask in range(1, 1 << 10):
        size = 0
        residue = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                size += 1
                residue += inv[denom]
                terms.append(denom)
        residue %= 101
        if size < min_size[residue]:
            min_size[residue] = size
            best[residue] = terms
        
        # Check deadline (should not trigger, but just in case)
        if time.time() > deadline:
            break
    
    # Construct result
    result = []
    for r in range(101):
        if best[r] is not None:
            result.append(best[r])
        else:
            # Fallback: find any subset that sums to r (should not happen)
            # This is a safety net
            result.append([])
    
    return result
