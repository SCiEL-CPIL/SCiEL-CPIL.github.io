"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    best_max_denom = float('inf')
    best_solution = None
    
    # Target fraction 5/31
    num, den = 5, 31
    
    # We search for solutions of the form 1/a + 1/b + 1/c = 5/31 with a < b < c
    # a must be at least ceil(31/5) = 7
    # If we find a 2-term solution, it would be 1/a + 1/b = 5/31
    # (5a-31)(5b-31) = 31^2 = 961
    # Divisors of 961: 1, 31, 961
    # Pairs (X, Y) with XY=961, X<Y:
    # 1, 961 -> 5a-31=1 => 5a=32 (no int)
    # 31, 31 -> 5a-31=31 => 5a=62 (no int)
    # So no 2-term solution.
    
    # Search 3-term solutions
    # a < b < c
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
    # So a is in [7, 18]
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
            
        # Remainder R = 5/31 - 1/a = (5a - 31) / (31a)
        n_rem = 5 * a - 31
        d_rem = 31 * a
        
        if n_rem <= 0:
            continue
            
        # Reduce fraction
        g = math.gcd(n_rem, d_rem)
        n_rem //= g
        d_rem //= g
        
        # We need 1/b + 1/c = n_rem / d_rem
        # (n_rem * b - d_rem) * (n_rem * c - d_rem) = d_rem^2
        # Let X = n_rem * b - d_rem, Y = n_rem * c - d_rem
        # X * Y = d_rem^2
        # b = (X + d_rem) / n_rem
        # c = (Y + d_rem) / n_rem
        # Conditions:
        # 1. b, c are integers
        # 2. a < b < c
        # 3. X < Y (to ensure b < c)
        # 4. X, Y have same sign (product positive)
        #    Since b, c > 0, n_rem*b - d_rem can be negative.
        #    However, 1/b < n_rem/d_rem => b > d_rem/n_rem.
        #    So n_rem*b - d_rem > 0.
        #    So X, Y must be positive.
        
        # Find divisors X of d_rem^2 such that X < d_rem (since X < Y and XY=d_rem^2 => X < d_rem)
        # Actually X < Y implies X < sqrt(d_rem^2) = d_rem.
        
        # Iterate through divisors of d_rem^2
        # To be efficient, iterate through divisors of d_rem, or just range up to sqrt(d_rem^2)
        # d_rem is at most 31*18 = 558. d_rem^2 ~ 300k.
        # Iterating up to 558 is very fast.
        
        d_rem_sq = d_rem * d_rem
        
        for x in range(1, d_rem + 1):
            if d_rem_sq % x != 0:
                continue
                
            y = d_rem_sq // x
            
            # Check if b and c are integers
            # b = (x + d_rem) / n_rem
            # c = (y + d_rem) / n_rem
            
            if (x + d_rem) % n_rem != 0:
                continue
            if (y + d_rem) % n_rem != 0:
                continue
                
            b = (x + d_rem) // n_rem
            c = (y + d_rem) // n_rem
            
            # Check order and distinctness
            if a < b < c:
                max_denom = c
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = [a, b, c]
                    
                    # If we found a very small one, we might stop early?
                    # Theoretical min for 3 terms?
                    # If best is 434, can we go lower?
                    # We will just let the loop finish, it's fast.
                    
    # If no 3-term solution found (unlikely), try 4 terms?
    # But 3-term is likely optimal for this size.
    # The seed had 3 terms.
    
    if best_solution is None:
        # Fallback to seed logic
        from fractions import Fraction
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return best_solution
# EVOLVE-BLOCK-END
