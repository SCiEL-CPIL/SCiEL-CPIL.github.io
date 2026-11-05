import time

_START = time.time()
_DEADLINE = 80.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # best_size[r] = minimum subset size that sums to r mod MOD
    # best_mask[r] = the mask achieving that minimum size
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD
    
    # Iterate over all non-empty subsets (only 1023 total, so this is fast)
    # We can iterate by size to ensure we find the minimum size first, 
    # or just iterate all and keep the min. 
    # Iterating all and keeping min is simpler and fast enough.
    
    for mask in range(1, 1 << N):
        # Calculate sum of inverses for this mask
        s = 0
        size = 0
        for x in range(1, N + 1):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % MOD
                size += 1
        
        if size < best_size[s]:
            best_size[s] = size
            best_mask[s] = mask
    
    # Construct the result
    # row r contains the denominators from best_mask[r]
    result = []
    for r in range(MOD):
        m = best_mask[r]
        terms = []
        for x in range(1, N + 1):
            if m & (1 << (x - 1)):
                terms.append(x)
        result.append(terms)
        
    return result
