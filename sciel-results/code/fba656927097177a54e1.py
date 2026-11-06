"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    
    # We want to find distinct positive integers d1, d2, ..., dk (k >= 1)
    # such that sum(1/di) = 5/31, minimizing max(di).
    
    # First, check 1-term solutions (trivial, but let's be thorough)
    # 1/d = 5/31 => d = 31/5, not integer.
    
    # Check 2-term solutions: 1/a + 1/b = 5/31
    # 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31^2 = 961
    # 961 = 1 * 961 = 31 * 31
    # Factors of 961: 1, 31, 961 (since 31 is prime)
    # 5a - 31 must be a positive divisor of 961 (since a > 0, 5a > 31 for 1/a < 5/31)
    # Actually 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7.
    # 5a - 31 >= 5*7 - 31 = 4 > 0. Good.
    
    best_solution = None
    best_max_denom = float('inf')
    
    # 2-term solutions
    # (5a-31)(5b-31) = 961
    # Let x = 5a-31, y = 5b-31, xy = 961
    # a = (x+31)/5, b = (y+31)/5
    # x and y must be such that x+31 and y+31 are divisible by 5.
    # x = 961k, y = 961/k for k dividing 961
    # Divisors of 961: 1, 31, 961
    for k in [1, 31, 961]:
        x = k
        y = 961 // k
        if (x + 31) % 5 != 0 or (y + 31) % 5 != 0:
            continue
        a = (x + 31) // 5
        b = (y + 31) // 5
        if a == b:
            continue
        if a >= 1 and b >= 1:
            # Check they are distinct
            max_d = max(a, b)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = [a, b]
    
    # 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a must satisfy: 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    # Also 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        N = r.numerator
        D = r.denominator
        
        # 1/b + 1/c = N/D
        # b > D/N
        # 2/b > N/D => b < 2D/N
        b_min = max(a + 1, D // N + 1)
        b_max = (2 * D) // N
        
        for b in range(b_min, b_max + 1):
            K = b * N - D
            if K <= 0:
                continue
            g = math.gcd(K, b)
            if D % (K // g) != 0:
                continue
            c = (b * D) // K
            if c <= b:
                continue
            if c == a or c == b:
                continue
            if a == b or a == c:
                continue
            max_d = max(a, b, c)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = [a, b, c]
    
    # 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a >= 7, a <= ? 
    # 4/a > 5/31 => a < 124/5 = 24.8, so a <= 24
    for a in range(7, 25):
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # Now find 3-term solution for r1: 1/b + 1/c + 1/d = r1
        # b >= a+1
        # 3/b > r1 => b < 3/r1
        # 1/b < r1 => b > 1/r1
        if r1.numerator == 0:
            continue
        b_min = a + 1
        b_max = min((3 * r1.denominator) // r1.numerator, 10000000)
        if b_min > b_max:
            continue
        for b in range(b_min, b_max + 1):
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            if r2.numerator == 0:
                # We found a 3-term solution (a, b, c where c is infinity? No, r2=0 means we used only a and b)
                # Wait, if r2 == 0, then 1/a + 1/b = target, which we already checked.
                continue
            # 1/c + 1/d = r2
            N2 = r2.numerator
            D2 = r2.denominator
            c_min = max(b + 1, D2 // N2 + 1)
            c_max = (2 * D2) // N2
            if c_min > c_max:
                continue
            for c in range(c_min, c_max + 1):
                K = c * N2 - D2
                if K <= 0:
                    continue
                g = math.gcd(K, c)
                if D2 % (K // g) != 0:
                    continue
                d = (c * D2) // K
                if d <= c:
                    continue
                if d == a or d == b or d == c:
                    continue
                max_d = max(a, b, c, d)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_solution = [a, b, c, d]
    
    if best_solution:
        return best_solution
    
    # Fallback
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
