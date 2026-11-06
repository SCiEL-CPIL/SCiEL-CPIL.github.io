"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # y range: b/a < y < 2*b/a
            y_lo = b // a + 1
            y_hi = min(2 * b // a, BOUND)
            
            # Also y > x for distinctness
            if y_lo <= x:
                y_lo = x + 1
            
            if y_lo > y_hi:
                continue
            
            # Use gcd to find valid y values efficiently
            # We need (a*y - b) | b*y
            # Let d = gcd(a, b), a' = a/d, b' = b/d
            # Then (a*y - b) | b*y iff (a'*y - b') | b'*y (after dividing by d)
            # This is equivalent to (a'*y - b') | b'*(y - (a'*y - b'))/a' ... 
            # Actually: (a*y - b) | b*y
            # Let g = gcd(a, b), a = g*a1, b = g*b1
            # (g*a1*y - g*b1) | g*b1*y
            # g*(a1*y - b1) | g*b1*y
            # (a1*y - b1) | b1*y
            # (a1*y - b1) | b1*y - b1*(a1*y - b1) = b1^2
            # So we need (a1*y - b1) | b1^2
            # Let d = a1*y - b1, then d | b1^2 and d = a1*y - b1, so y = (d + b1)/a1
            # d must divide b1^2 and d ≡ -b1 (mod a1) and y > y_lo, y < y_hi
            
            g = math.gcd(a, b)
            a1 = a // g
            b1 = b // g
            
            # We need d | b1^2 and d ≡ -b1 (mod a1) and y = (d + b1)/a1 in [y_lo, y_hi]
            # So d must be in [(y_lo - b1)*a1, (y_hi - b1)*a1] and d | b1^2
            # and d ≡ -b1 (mod a1)
            
            d_lo = (y_lo - b1) * a1
            d_hi = (y_hi - b1) * a1
            
            if d_lo < 1:
                d_lo = 1
            
            # Find divisors of b1^2 in range [d_lo, d_hi] with d ≡ -b1 (mod a1)
            # b1^2 can be large, so we factor b1 and generate divisors
            # But b1 can be up to n*x ~ 256*192 ~ 50000, so b1^2 ~ 2.5e9
            # Let's try a different approach: iterate over possible d values
            
            # Actually, let's just iterate y but with a step based on a1
            # Since d = a1*y - b1 must divide b1^2, and d ≡ -b1 (mod a1)
            # Let's iterate over divisors of b1^2
            
            # Factor b1
            def get_divisors(n_val):
                divs = set()
                i = 1
                while i * i <= n_val:
                    if n_val % i == 0:
                        divs.add(i)
                        divs.add(n_val // i)
                    i += 1
                return sorted(divs)
            
            b1_sq = b1 * b1
            # Get divisors of b1^2
            divs = get_divisors(b1_sq)
            
            found_for_x = None
            for d in divs:
                if d < d_lo or d > d_hi:
                    continue
                # Check d ≡ -b1 (mod a1)
                if (d + b1) % a1 != 0:
                    continue
                y_val = (d + b1) // a1
                if y_val < y_lo or y_val > y_hi:
                    continue
                # Compute z
                z_val = b * y_val // (a * y_val - b)
                if z_val > y_val and z_val <= BOUND:
                    if x != y_val and y_val != z_val and x != z_val:
                        found_for_x = [x, y_val, z_val]
                        break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
