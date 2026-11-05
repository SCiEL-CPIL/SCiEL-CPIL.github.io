import time

_START = time.time()
_DEADLINE = 85.0

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
    # Initialize with a large number
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD
    
    # We can iterate through masks. 
    # To be slightly more efficient, we can compute sums incrementally 
    # or just stick to the simple loop which is fast enough.
    
    # Simple loop is O(2^N * N). 1024 * 10 = 10k ops. Trivial.
    
    for mask in range(1, 1 << N):
        if _elapsed() > _DEADLINE:
            # If we timeout, return best found so far
            # But we won't timeout.
            break
            
        s = 0
        size = 0
        m = mask
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
            
        r = s % MOD
        
        if size < best_size[r]:
            best_size[r] = size
            best_mask[r] = mask
            
    # Construct rows
    rows = []
    for r in range(MOD):
        if best_mask[r] == 0:
            # Fallback: This should not happen if coverage is complete.
            # If it does, we need a valid non-empty list.
            # We can just pick the first available denominator.
            rows.append([1])
        else:
            mask = best_mask[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows.append(terms)
            
    return rows
