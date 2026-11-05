def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # c = 31ab / (5ab - 31a - 31b)
    # We want to minimize max(a, b, c)
    
    # Try to beat current best of 1736
    upper_bound = 1736
    
    for a in range(1, upper_bound + 1):
        for b in range(a + 1, upper_bound + 1):
            D = 5 * a * b - 31 * a - 31 * b
            if D <= 0:
                continue
            num = 31 * a * b
            if num % D != 0:
                continue
            c = num // D
            if c <= 0 or c > 10000000:
                continue
            if c == a or c == b:
                continue
            # Verify with Fraction
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s != target:
                continue
            m = max(a, b, c)
            if m < best_max:
                best_max = m
                best = [a, b, c]
    
    if best is not None:
        return best
    
    # Fallback: use the original greedy approach
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
