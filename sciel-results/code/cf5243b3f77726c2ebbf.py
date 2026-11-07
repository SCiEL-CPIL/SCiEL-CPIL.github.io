"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Primes relevant to 210 = 2*3*5*7
    primes = [2, 3, 5, 7]
    
    def sf_mask(n):
        """Return bitmask of which primes from {2,3,5,7} appear in sf(n)"""
        mask = 0
        for idx, p in enumerate(primes):
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << idx)
        return mask, n  # return mask and remaining part
    
    def sf_full(n):
        """Full squarefree kernel"""
        res = 1
        i = 2
        while i * i <= n:
            cnt = 0
            while n % i == 0:
                n //= i
                cnt += 1
            if cnt % 2 == 1:
                res *= i
            i += 1
        if n > 1:
            res *= n
        return res
    
    lower_bound = 211
    upper_bound = 10210
    
    # Precompute sf masks for all candidates
    # mask[n] = bitmask of primes in {2,3,5,7} that appear in sf(n)
    # Also store the "other" part of sf(n) not in {2,3,5,7}
    masks = {}
    other_parts = {}
    
    for n in range(lower_bound, upper_bound + 1):
        mask, remaining = sf_mask(n)
        # The remaining part may have other primes
        # We need to check if remaining has any prime factors with odd exponent
        other_sf = 1
        i = 11  # Start after 7
        while i * i <= remaining:
            cnt = 0
            while remaining % i == 0:
                remaining //= i
                cnt += 1
            if cnt % 2 == 1:
                other_sf *= i
            i += 1
        if remaining > 1:
            other_sf *= remaining
        masks[n] = mask
        other_parts[n] = other_sf
    
    # Target: sf(product * 210) == 1
    # 210 has mask = 1111 (all four primes)
    # For the product to have sf=1, we need:
    # - The XOR of all masks (including 210's mask=15) must be 0
    # - All "other" parts must multiply to a perfect square
    
    # Actually, let's think more carefully:
    # sf(a*b*210) = 1 means a*b*210 is a perfect square
    # This means for every prime p, the total exponent in a*b*210 is even
    
    # For primes 2,3,5,7: the sum of exponents from a, b, and 210 must be even
    # 210 contributes 1 to each of 2,3,5,7
    # So for each p in {2,3,5,7}: exponent_in(a) + exponent_in(b) + 1 must be even
    # i.e., exponent_in(a) + exponent_in(b) must be odd
    # i.e., exactly one of a,b has odd exponent for p
    
    # For other primes: exponent_in(a) + exponent_in(b) must be even
    # i.e., same parity for a and b
    
    # This is getting complex. Let me just use the direct approach with optimization.
    
    # Key insight: For size 1, we need sf(n) = 210
    # n = 210 * k^2 for some k >= 1
    # n > 210, so k >= 1, n = 210, 840, 1890, ...
    # Smallest n > 210 with sf(n) = 210 is 840
    
    # For size 2, let's search for minimum max
    best_solution = None
    best_max = upper_bound + 1
    
    # Size 1 check
    for k in range(1, 100):
        n = 210 * k * k
        if n > upper_bound:
            break
        if n > lower_bound:
            best_solution = [n]
            best_max = n
            break
    
    # Size 2 check: search for minimum max
    if best_max > 2:
        # For each possible max M, check if there's a valid pair
        # Optimize: precompute for each number its "signature"
        # Two numbers a, b form a valid pair if sf(a*b*210) = 1
        
        # Let's just iterate M from lower_bound+1 upward
        for M in range(lower_bound + 1, min(upper_bound + 1, 500)):
            if time.time() > deadline:
                break
            if M >= best_max:
                break
            # Check all pairs (a, b) with a < b <= M
            found = False
            for b in range(M, lower_bound, -1):
                if time.time() > deadline or found:
                    break
                for a in range(b - 1, lower_bound - 1, -1):
                    # Check if sf(a * b * 210) == 1
                    prod = a * b * 210
                    root = math.isqrt(prod)
                    if root * root == prod:
                        best_solution = [a, b]
                        best_max = M
                        found = True
                        break
            if found:
                break
    
    # Size 3 check
    if best_max > 3:
        for M in range(lower_bound + 2, min(upper_bound + 1, 300)):
            if time.time() > deadline:
                break
            if M >= best_max:
                break
            found = False
            for c in range(M, lower_bound + 1, -1):
                if time.time() > deadline or found:
                    break
                for b in range(c - 1, lower_bound, -1):
                    if time.time() > deadline or found:
                        break
                    for a in range(b - 1, lower_bound - 1, -1):
                        prod = a * b * c * 210
                        root = math.isqrt(prod)
                        if root * root == prod:
                            best_solution = [a, b, c]
                            best_max = M
                            found = True
                            break
            if found:
                break
    
    # Fallback
    if best_solution is None:
        best_solution = [224, 240]
    
    return best_solution
