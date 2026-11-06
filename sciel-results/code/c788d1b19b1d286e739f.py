def construct():
    import time
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    deadline = time.time() + 80  # Safety margin for 90s limit
    
    best_max = float('inf')
    best_solution = None
    
    def try_update(solution):
        nonlocal best_max, best_solution
        m = max(solution)
        if m < best_max:
            best_max = m
            best_solution = list(solution)
    
    # 3-term search: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a in [7, 18]
    for a in range(7, 19):
        if time.time() > deadline:
            break
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        n = r.numerator
        d = r.denominator
        # (nb - d)(nc - d) = d^2
        d2 = d * d
        # Find all divisor pairs (X, Y) with X*Y = d2, X <= Y
        # b = (X + d) / n must be integer, c = (Y + d) / n must be integer
        # b > a, c > b
        
        # Iterate X from 1 to sqrt(d2)
        sqrt_d2 = int(math.isqrt(d2))
        for X in range(1, sqrt_d2 + 1):
            if d2 % X != 0:
                continue
            Y = d2 // X
            # Try both orderings: (X, Y) and (Y, X)
            for X1, Y1 in [(X, Y), (Y, X)]:
                if (X1 + d) % n != 0:
                    continue
                b = (X1 + d) // n
                if (Y1 + d) % n != 0:
                    continue
                c = (Y1 + d) // n
                if b <= a or c <= b:
                    continue
                if c > 10000000:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    try_update([a, b, c])
    
    # 4-term search: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # Only search if we can potentially improve best_max
    # a in [7, 24] (since 4/a > 5/31 => a < 124/5 = 24.8)
    for a in range(7, 25):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        n1 = r1.numerator
        d1 = r1.denominator
        # 1/b + 1/c + 1/d = r1
        # b > a, and 3/b > r1 => b < 3/r1
        b_max = int(3 / r1)
        if b_max > 10000000:
            b_max = 10000000
        # Also b must be such that remaining can be completed
        # To keep search manageable, limit b range
        b_upper = min(b_max, a + 200)  # Practical limit
        
        for b in range(a + 1, b_upper + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            n2 = r2.numerator
            d2 = r2.denominator
            # 1/c + 1/d = r2
            # (n2*c - d2)(n2*d - d2) = d2^2
            d2_sq = d2 * d2
            sqrt_d2_sq = int(math.isqrt(d2_sq))
            
            # c > b, and 2/c > r2 => c < 2/r2
            c_max = int(2 / r2)
            if c_max > 10000000:
                c_max = 10000000
            
            for X in range(1, sqrt_d2_sq + 1):
                if d2_sq % X != 0:
                    continue
                Y = d2_sq // X
                for X1, Y1 in [(X, Y), (Y, X)]:
                    if (X1 + d2) % n2 != 0:
                        continue
                    c = (X1 + d2) // n2
                    if (Y1 + d2) % n2 != 0:
                        continue
                    d = (Y1 + d2) // n2
                    if c <= b or d <= c:
                        continue
                    if d > 10000000:
                        continue
                    # Check if this improves best_max
                    if d >= best_max:
                        continue
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                        try_update([a, b, c, d])
    
    # Fallback
    if best_solution is None:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values
    
    return best_solution
