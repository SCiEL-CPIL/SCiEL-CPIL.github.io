"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    target = Fraction(5, 31)
    
    # Fallback: The known greedy solution
    # 5/31 - 1/8 = 9/248
    # 9/248 - 1/28 = 1/1736
    fallback = [8, 28, 1736]
    best_max = 1736
    
    # Search for 3-term solutions with max denominator < 1736
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # c < 1736
    # 1/a + 1/b > 5/31 - 1/1736
    # 1/a < 5/31
    
    max_d = 1735
    
    # Iterate a
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also 3/a > 5/31 => a < 3*31/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        # 1/b + 1/c = rem_a
        # 1/b < rem_a => b > 1/rem_a
        # 2/b > rem_a => b < 2/rem_a
        min_b = (1 // rem_a) + 1
        max_b = (2 // rem_a)
        
        # Also b > a
        if min_b <= a:
            min_b = a + 1
            
        # c < max_d => 1/c > 1/max_d
        # 1/b - 1/max_d < rem_a => 1/b < rem_a + 1/max_d
        # b > 1 / (rem_a + 1/max_d)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b <= a:
                continue
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            
            # c = 1 / rem_b
            if rem_b.denominator % rem_b.numerator == 0:
                c = rem_b.denominator // rem_b.numerator
                if c > b and c <= max_d:
                    # Valid solution
                    if c < best_max:
                        best_max = c
                        fallback = [a, b, c]
                    # We can break early if we find a very small one?
                    # Just keep searching for even smaller?
                    # Or break if we find one? The goal is minimize max.
                    # Since c is decreasing as b increases (in this range?), 
                    # actually c = 1/rem_b. 
                    # rem_b = rem_a - 1/b. As b increases, 1/b decreases, rem_b increases.
                    # So c decreases.
                    # So the last valid b gives the smallest c.
                    # But we must ensure c is integer.
                    pass
                    
    # If no 3-term solution found with max < 1736, try 4-term?
    # 4-term search is more expensive.
    # Given the fallback is 1736, and we likely won't find a 3-term solution 
    # with max < 1736 easily (none found in quick manual check for a=9),
    # the fallback is likely the best or close to it.
    
    # However, the knowledge suggests scores of 7-9 are possible.
    # This implies max denom ~ 200.
    # This requires many terms.
    # A simple greedy completion from a small start might work?
    # Try starting with 1/7, 1/56...
    # But we already checked 1/7, 1/56 -> 1/1736.
    
    # Let's try a quick greedy from different starts to see if we get lucky.
    starts = [7, 8, 9, 10, 11, 12]
    for s in starts:
        if time.time() > deadline:
            break
        r = target - Fraction(1, s)
        vals = [s]
        while r > 0:
            if time.time() > deadline:
                break
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d in vals:
                break
            if d > 10000000:
                break
            vals.append(d)
            r -= Fraction(1, d)
            if r == 0:
                max_v = max(vals)
                if max_v < best_max:
                    best_max = max_v
                    fallback = vals
                break
    
    return fallback
# EVOLVE-BLOCK-END
