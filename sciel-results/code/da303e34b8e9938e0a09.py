import time
from fractions import Fraction
from math import gcd

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def check_time():
        return time.time() > deadline
    
    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Verify
    s = Fraction(1,8) + Fraction(1,31) + Fraction(1,248)
    assert s == TARGET, f"Known solution invalid: {s}"
    
    # Number-theoretic approach for 3-term solutions:
    # 1/a + 1/b + 1/c = 5/31 with 7 <= a < b < c
    # For fixed c, we need 1/a + 1/b = 5/31 - 1/c = (5c-31)/(31c)
    # So (a+b)/(ab) = (5c-31)/(31c)
    # => 31c(a+b) = ab(5c-31)
    # Let K = 5c - 31. Then: Kab - 31ca - 31cb = 0
    # => a(Kb - 31c) = 31c(b - c)
    # => a = 31c(b-c) / (Kb - 31c)
    #
    # For a to be positive integer with a < b < c:
    # We need Kb - 31c > 0, i.e., b > 31c/K = 31c/(5c-31)
    # And Kb - 31c must divide 31c(b-c)
    #
    # Alternative: enumerate divisors. Let u = Kb - 31c.
    # Then b = (u + 31c)/K, and we need u + 31c to be divisible by K.
    # Also a = 31c(b-c)/u = 31c((u+31c)/K - c)/u = 31c(u + 31c - cK)/(Ku)
    # = 31c(u + 31c - c(5c-31))/(Ku) = 31c(u + 31c - 5c^2 + 31c)/(Ku)
    # = 31c(u + 62c - 5c^2)/(Ku)
    #
    # Let N = 62c - 5c^2 = c(62 - 5c). For c > 12.4, N < 0.
    # For c >= 13, N < 0. For c <= 12, N > 0.
    # But we need a < b < c and a >= 7, so c >= 9 at minimum.
    #
    # Actually, let me just iterate c from 9 to best_max-1, and for each c,
    # iterate over possible divisors to find valid (a,b) pairs.
    
    # Simpler approach: for each c, iterate b from max(a+1, floor(31c/(5c-31))+1) to c-1,
    # and check if a = 31c(b-c)/(Kb-31c) is a positive integer with a < b.
    
    # But that's still O(c^2) per c. Let me use the divisor enumeration.
    
    # From a = 31c(b-c)/(Kb-31c), let d = Kb - 31c.
    # Then b = (d + 31c)/K, and a = 31c((d+31c)/K - c)/d = 31c(d + 31c - cK)/(Kd)
    # = 31c(d + 31c - c(5c-31))/(Kd) = 31c(d + 62c - 5c^2)/(Kd)
    
    # Let P = 31c(62c - 5c^2) = 31c^2(62 - 5c)
    # Then a = (31c * d + P) / (Kd) = 31c/K + P/(Kd)
    
    # For a to be integer, Kd must divide 31c*d + P, i.e., Kd divides P.
    # So d must be a divisor of P / gcd(P, K) ... this is getting complicated.
    
    # Let me just do the direct enumeration but more efficiently.
    # For 3-term: iterate c from 9 to best_max-1
    # For each c, K = 5c - 31
    # b must satisfy: b > 31c/K (so that a > 0) and b < c
    # And a = 31c(b-c)/(Kb-31c) must be a positive integer < b
    
    # The range of b: b_min = floor(31c/K) + 1, b_max = c - 1
    # If b_min >= b_max, skip
    
    # Let me also try 4-term solutions for smaller max values.
    
    # First, check all 3-term solutions with max < 248
    for c in range(9, best_max):
        if check_time():
            break
        K = 5 * c - 31
        if K <= 0:
            continue
        # b > 31c/K
        b_min = (31 * c) // K + 1
        if b_min < 8:
            b_min = 8
        b_max = c - 1
        if b_min > b_max:
            continue
        for b in range(b_min, b_max + 1):
            if check_time():
                break
            denom = K * b - 31 * c
            if denom <= 0:
                continue
            num = 31 * c * (b - c)
            # a = num / denom
            if num % denom != 0:
                continue
            a = num // denom
            if a < 7 or a >= b:
                continue
            # Valid solution: a, b, c with max = c
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
    
    # Now try 4-term solutions for even smaller max
    # 1/a + 1/b + 1/c + 1/d = 5/31 with a < b < c < d
    # For fixed d, need 1/a + 1/b + 1/c = 5/31 - 1/d
    # This is a 3-term problem for each d, which we can solve similarly.
    
    # But first, let's see if we found a 3-term solution better than 248
    # If best_max is still 248, try 4-term solutions with max < 248
    
    if best_max == 248:
        # Try 4-term solutions with max < 248
        # For d from 10 to 247, solve 1/a + 1/b + 1/c = 5/31 - 1/d
        for d in range(10, 248):
            if check_time():
                break
            target_3 = TARGET - Fraction(1, d)
            if target_3 <= 0:
                continue
            # Solve 1/a + 1/b + 1/c = target_3 with a < b < c < d
            # For fixed c, 1/a + 1/b = target_3 - 1/c
            # Let's iterate c from 8 to d-1
            for c in range(8, d):
                if check_time():
                    break
                rem_2 = target_3 - Fraction(1, c)
                if rem_2 <= 0:
                    continue
                # 1/a + 1/b = rem_2 with a < b < c
                # (a+b)/(ab) = rem_2
                # Let rem_2 = p/q in lowest terms
                p = rem_2.numerator
                q = rem_2.denominator
                # q(a+b) = p*ab
                # pa*b - q*a - q*b = 0
                # a(pb - q) = q(b - q/p)... 
                # Let's use: a = q(b - q/p) / (pb - q)... 
                # Better: pab - qa - qb = 0
                # a(pb - q) = qb
                # a = qb / (pb - q)
                # For a to be positive integer: pb - q > 0, i.e., b > q/p
                # And pb - q must divide qb
                
                # b > q/p
                b_min_2 = (q + p - 1) // p + 1  # ceil(q/p) + 1... wait
                # b > q/p means b >= floor(q/p) + 1
                b_min_2 = q // p + 1
                if b_min_2 < 8:
                    b_min_2 = 8
                b_max_2 = c - 1
                if b_min_2 > b_max_2:
                    continue
                for b in range(b_min_2, b_max_2 + 1):
                    if check_time():
                        break
                    denom2 = p * b - q
                    if denom2 <= 0:
                        continue
                    num2 = q * b
                    if num2 % denom2 != 0:
                        continue
                    a = num2 // denom2
                    if a < 7 or a >= b:
                        continue
                    # Valid 4-term solution: a, b, c, d with max = d
                    if d < best_max:
                        best_max = d
                        best_solution = [a, b, c, d]
    
    return best_solution
