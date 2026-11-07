import time
import math
from functools import lru_cache

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of 1..12
_LCM = 1
for i in range(1, 13):
    _LCM = _lcm(_LCM, i)
# _LCM = 27720

N = 12
MAX_L = 256  # Safe upper bound for L

# Precompute masks for each i in 1..12 and each residue r in 0..i-1
# mask[i][r] is a bitmask of length MAX_L where bit k is 1 if k % i == r
_precomputed_masks = [[0] * i for i in range(N + 1)]
for i in range(1, N + 1):
    for r in range(i):
        mask = 0
        for k in range(MAX_L):
            if k % i == r:
                mask |= (1 << k)
        _precomputed_masks[i][r] = mask

def _can_match(m, L):
    """
    Check if (m, m+L] contains distinct b_1...b_N with i | b_i.
    """
    limit = (1 << L) - 1
    masks = []
    for i in range(1, N + 1):
        r = (-m) % i
        mask = _precomputed_masks[i][r] & limit
        if mask == 0:
            return False
        masks.append(mask)
    
    # Sort masks by popcount (most constrained first)
    sorted_masks = sorted(masks, key=lambda x: x.bit_count())
    
    @lru_cache(maxsize=None)
    def dfs(k, used):
        if k == N:
            return True
        mask = sorted_masks[k]
        avail = mask & ~used
        while avail:
            bit = avail & -avail
            if dfs(k + 1, used | bit):
                return True
            avail -= bit
        return False
    
    return dfs(0, 0)

def _f_val(m):
    """
    Find the least positive L such that (m, m+L] contains distinct b_1...b_N with i | b_i.
    """
    # Linear search from 12 upwards is likely faster than binary search
    # because the optimal L is usually small.
    for L in range(N, MAX_L + 1):
        if _can_match(m, L):
            return L
    return MAX_L  # Fallback

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 seconds, leaving 5s margin
    
    best_f = 0
    best_m = 0
    
    # Iterate over the fundamental domain of the periodic function
    for m in range(_LCM):
        if time.time() > deadline:
            break
        
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    return int(best_m)
