"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    def try_assign(L, W):
        """Try to assign distinct multiples in [L, L+W] for each divisor."""
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d  # ceil(L/d)
            k_max = (L + W) // d       # floor((L+W)/d)
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
        
        used = set()
        assignment = [None] * n
        
        def backtrack(i):
            if i == n:
                return True
            # Optimization: sort by number of choices? Or just standard.
            # Standard is fine for small N.
            for m in avail[i]:
                if m not in used:
                    used.add(m)
                    assignment[i] = m
                    if backtrack(i + 1):
                        return True
                    used.remove(m)
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Search for minimum W
    # Minimum W is 9 because we need 10 distinct integers.
    # We search L in a reasonable range.
    # The pattern of valid L repeats with period LCM, but we assume a solution
    # exists in the first few thousand integers if W is small enough.
    # If not found in 0..5000, we expand or use fallback.
    
    search_range = 5000
    
    for W in range(9, 100):
        for L in range(1, search_range + 1):
            if L + W > max_val:
                break
            
            # Quick check: Does each divisor have a multiple in [L, L+W]?
            valid = True
            for d in divisors:
                # Check if there is a multiple of d in [L, L+W]
                # First multiple >= L is L + (d - L % d) % d
                offset = (d - L % d) % d
                if offset > W:
                    valid = False
                    break
            
            if not valid:
                continue
            
            result = try_assign(L, W)
            if result is not None:
                # Verify
                assert all(isinstance(x, int) for x in result)
                assert len(set(result)) == n
                assert all(result[i] % divisors[i] == 0 for i in range(n))
                assert all(1 <= x <= max_val for x in result)
                return list(result)
    
    # Fallback: return the divisors themselves
    return divisors[:]
# EVOLVE-BLOCK-END
