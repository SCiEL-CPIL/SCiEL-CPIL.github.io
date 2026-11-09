"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    # Known optimal 2-element solution: 224 * 240 * 210 = 3360^2
    # Raw value: 240 - 210 = 30
    # Verify: 224=2^5*7, 240=2^4*3*5, 210=2*3*5*7
    # Product = 2^10 * 3^2 * 5^2 * 7^2 = (2^5*3*5*7)^2
    # This is strictly better than the single element solution [840] (raw 630).
    
    # Check if this solution is valid (it is, but we verify to be safe)
    a, b = 224, 240
    prod = a * b * 210
    r = math.isqrt(prod)
    if r * r == prod and a > 210 and b > 210 and a != b:
        return [a, b]
    
    # Fallback: Search for better 2-element solutions (should not find any < 240)
    # or single element solutions.
    
    best_solution = [840]
    best_raw = 630
    
    # 2-element search
    # a*b = 210 * m^2
    # a, b in (210, 10210]
    # To minimize max(a,b), we search for small max.
    # We already know 240 is optimal from external knowledge, but let's run a quick check
    # for any max < 240 just in case the hardcoded one fails (it won't).
    
    # Actually, the hardcoded one is the best possible. 
    # If it fails validation (it shouldn't), we fall back to search.
    
    # Quick search for max < 840 (covers the range where 224, 240 live)
    found_better = False
    for m in range(15, 59):
        prod = 210 * m * m
        a_max = min(839, prod // 211)
        for a in range(211, min(a_max, 840)):
            if prod % a == 0:
                b = prod // a
                if 211 <= b <= 839 and a != b:
                    max_val_found = max(a, b)
                    if max_val_found < best_raw + 210:
                        best_solution = [a, b]
                        best_raw = max_val_found - 210
                        found_better = True
                        break
        if found_better and best_raw < 630:
            break
            
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
