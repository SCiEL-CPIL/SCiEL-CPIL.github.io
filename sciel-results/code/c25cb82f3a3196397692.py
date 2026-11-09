def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = [8, 31, 248] # Fallback
    
    # Search for 3-term solutions 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # 1/a < 5/31 => a > 6.2 => a >= 7
    # 3/a > 5/31 => a < 18.6 => a <= 18
    
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        # 1/b + 1/c = r = num/den
        num, den = r.numerator, r.denominator
        # num * b * c = den * (b + c)
        # num^2 * b * c - num * den * b - num * den * c = 0
        # (num * b - den) * (num * c - den) = den^2
        # Let X = num * b - den, Y = num * c - den
        # X * Y = den^2
        # b = (X + den) / num must be integer
        # c = (Y + den) / num must be integer
        # b >= a+1, c > b
        
        den_sq = den * den
        # Iterate divisors X of den_sq
        # X <= Y => X <= sqrt(den_sq) = den
        # Also b >= a+1 => (X+den)/num >= a+1 => X >= num(a+1) - den
        min_X = max(1, num * (a + 1) - den)
        
        for X in range(min_X, int(math.isqrt(den_sq)) + 1):
            if den_sq % X != 0:
                continue
            Y = den_sq // X
            if (X + den) % num != 0:
                continue
            b = (X + den) // num
            if (Y + den) % num != 0:
                continue
            c = (Y + den) // num
            
            if b < a + 1:
                continue
            if c <= b:
                continue
            
            # Verify distinct and sum
            # a, b, c are distinct by construction (b>a, c>b)
            # Check sum
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                max_d = c
                if max_d < best_max:
                    best_max = max_d
                    best_solution = [a, b, c]
                    
    return best_solution
