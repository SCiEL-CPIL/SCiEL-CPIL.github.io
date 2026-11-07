"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    # For each i in 1..n, find which positions j in 0..L-1 have (m+1+j) % i == 0
    # i.e., (m+1+j) is divisible by i
    # j % i == (-(m+1)) % i
    
    options = []
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        for j in range(L):
            if j % i == target:
                mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    n_opts = n
    
    # DFS with bitmask
    def dfs(k, used_mask):
        if k == n_opts:
            return True
        cnt, mask = options[k]
        avail = mask & ~used_mask
        while avail:
            bit = avail & (-avail)
            avail -= bit
            if dfs(k + 1, used_mask | bit):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] contains distinct b_1...b_n with i | b_i.
    Binary search on L.
    """
    # Lower bound: at least n (need n distinct numbers)
    # Upper bound: something reasonable, say n * n or larger
    lo = 1
    hi = n * n  # Start with a small upper bound, expand if needed
    
    # First, find a valid hi
    while not _can_match(m, hi, n):
        hi *= 2
        if hi > 10000:
            break
    
    # Binary search for the minimum L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 seconds, leaving 10s margin
    
    n = 12
    max_m = 1000000
    lcm_period = _LCM  # 27720
    
    # We'll evaluate f(12, m) for a set of candidate m values.
    # Since f depends on m mod lcm_period, we can group by residue.
    # But we still need to check which specific m in [0, max_m] gives the best f.
    
    # Strategy: 
    # 1. For each residue r in 0..lcm_period-1, the best m with that residue is the smallest m in [0, max_m] with m % lcm_period == r.
    #    Actually, f(12, m) depends on m mod lcm_period, so all m with the same residue give the same f.
    # 2. So we just need to find the residue r that maximizes f(12, r) (or f(12, m) for any m with m % lcm_period == r).
    
    # Let's compute f for each residue class representative.
    # Representative for residue r: m = r (if r <= max_m, which it always is since r < 27720 < 1000000)
    
    best_f = 0
    best_m = 0
    best_residue = 0
    
    # We'll iterate over residues, but check time periodically
    for r in range(lcm_period):
        if time.time() > deadline:
            break
        
        f = _f_val(r, n)
        if f > best_f:
            best_f = f
            best_m = r
            best_residue = r
    
    # Now, among all m in [0, max_m] with m % lcm_period == best_residue, pick one.
    # The smallest such m is best_residue itself (since best_residue < lcm_period <= max_m).
    # But we could also pick any other m with the same residue. The value f is the same.
    # So just return best_residue.
    
    # Actually, let's double-check: is best_residue <= max_m? Yes, since best_residue < 27720 < 1000000.
    
    return int(best_m)

# EVOLVE-BLOCK-END
