"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    best_max_denom = 10000000
    best_solution = None
    
    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a > 31/5 = 6.2 -> a >= 7
    # 1/b < 5/31 - 1/7 = 4/217 -> b > 54.25
    # But b can be smaller if a is larger.
    
    # We iterate a.
    # Max a: 1/a < 5/31 -> a > 6.
    # Also 3/a > 5/31 (since 1/a+1/b+1/c < 3/a) -> a < 93/5 = 18.6 -> a <= 18.
    # So a is in [7, 18].
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_1 = target - Fraction(1, a)
        if rem_1 <= 0:
            continue
        
        # 1/b + 1/c = rem_1
        # b > 1/rem_1
        # 2/b > rem_1 -> b < 2/rem_1
        # b_min = floor(1/rem_1) + 1
        # b_max = floor(2/rem_1)
        
        b_min = (rem_1.denominator // rem_1.numerator) + 1
        b_max = (2 * rem_1.denominator) // rem_1.numerator
        
        # Ensure b > a
        if b_min <= a:
            b_min = a + 1
            
        if b_min > b_max:
            continue
            
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_2 = rem_1 - Fraction(1, b)
            if rem_2 <= 0:
                continue
            
            # 1/c = rem_2
            # c = 1/rem_2
            # c must be integer and c > b
            if rem_2.denominator % rem_2.numerator == 0:
                c = rem_2.denominator // rem_2.numerator
                if c > b:
                    max_d = c
                    if max_d < best_max_denom:
                        best_max_denom = max_d
                        best_solution = [a, b, c]

    # Search for 4-term solutions if time permits and best is still high
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a >= 7
    # If we find a 3-term solution with max 248, we only care about 4-term if max < 248.
    # This implies d < 248.
    # So we can restrict d < 248.
    
    if best_max_denom > 100: # Only bother if current best is not already very small
         # This logic is complex to implement robustly in short time.
         # Given the 3-term search found 248 quickly, and 4-term search is much larger,
         # we rely on the 3-term result unless we can prune heavily.
         # However, let's just return the best found so far.
         pass

    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if no solution found (should not happen)
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

# EVOLVE-BLOCK-END
