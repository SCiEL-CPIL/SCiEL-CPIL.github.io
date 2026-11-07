"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of all divisors
LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, each divisible by its corresponding
    divisor in DIVISORS, minimizing max - min.
    
    Uses bitset-based feasibility checking over window start positions modulo LCM.
    """
    
    def is_feasible_fast(R, S_mod):
        """
        Check if window [S, S+R] with S ≡ S_mod (mod LCM) has a valid assignment
        using bitset operations for speed.
        """
        # For each divisor i, compute bitmask of valid k in [0, R]
        # k is valid if (S_mod + k) % DIVISORS[i] == 0
        # i.e., k % DIVISORS[i] == (-S_mod) % DIVISORS[i]
        
        bitmasks = []
        for i, d in enumerate(DIVISORS):
            rem = (-S_mod) % d
            mask = 0
            k = rem
            while k <= R:
                mask |= (1 << k)
                k += d
            if mask == 0:
                return False
            bitmasks.append(mask)
        
        # Now check if we can assign distinct k values using backtracking
        # Order by number of available k values (popcount) for efficiency
        def popcount(x):
            count = 0
            while x:
                x &= x - 1
                count += 1
            return count
        
        order = sorted(range(N), key=lambda i: popcount(bitmasks[i]))
        
        used_mask = 0
        
        def backtrack(idx):
            nonlocal used_mask
            if idx == N:
                return True
            i = order[idx]
            avail = bitmasks[i] & ~used_mask
            while avail:
                # Pick lowest set bit
                lb = avail & (-avail)
                avail ^= lb
                used_mask |= lb
                if backtrack(idx + 1):
                    return True
                used_mask ^= lb
            return False
        
        return backtrack(0)
    
    # Search for minimum R starting from 9 (need at least 10 distinct values in window)
    for R in range(9, 200):
        for S_mod in range(LCM):
            if is_feasible_fast(R, S_mod):
                # Found feasible window. Find actual assignment.
                # Determine valid S: S = S_mod + t*LCM, S >= 1, S+R <= 1000000
                if S_mod >= 1:
                    S = S_mod
                else:
                    t_min = (1 - S_mod + LCM - 1) // LCM
                    S = S_mod + t_min * LCM
                
                if S + R > 1000000:
                    continue
                
                # Reconstruct assignment
                bitmasks = []
                for i, d in enumerate(DIVISORS):
                    rem = (-S_mod) % d
                    mask = 0
                    k = rem
                    while k <= R:
                        mask |= (1 << k)
                        k += d
                    bitmasks.append(mask)
                
                def popcount(x):
                    count = 0
                    while x:
                        x &= x - 1
                        count += 1
                    return count
                
                order = sorted(range(N), key=lambda i: popcount(bitmasks[i]))
                used_mask = 0
                assignment = [0] * N
                
                def backtrack2(idx):
                    nonlocal used_mask
                    if idx == N:
                        return True
                    i = order[idx]
                    avail = bitmasks[i] & ~used_mask
                    while avail:
                        lb = avail & (-avail)
                        avail ^= lb
                        # Extract k from lb
                        k = lb.bit_length() - 1
                        used_mask |= lb
                        assignment[i] = k
                        if backtrack2(idx + 1):
                            return True
                        used_mask ^= lb
                    return False
                
                if backtrack2(0):
                    result = [S + assignment[i] for i in range(N)]
                    # Verify
                    assert all(isinstance(x, int) for x in result)
                    assert len(set(result)) == N
                    assert all(1 <= x <= 1000000 for x in result)
                    assert all(result[i] % DIVISORS[i] == 0 for i in range(N))
                    return list(result)
    
    # Fallback
    return DIVISORS[:]

# EVOLVE-BLOCK-END
