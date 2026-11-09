def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    
    best_max_den = float('inf')
    best_solution = None
    
    def try_solution(sol):
        nonlocal best_max_den, best_solution
        mx = max(sol)
        if mx < best_max_den:
            best_max_den = mx
            best_solution = sol
    
    # 3-term search
    for a in range(7, 19):
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        lower_b = max(a + 1, rem_a.denominator // rem_a.numerator + 1)
        upper_b = (2 * rem_a.denominator) // rem_a.numerator
        for b in range(lower_b, upper_b + 1):
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            if rem_b.numerator == 1:
                c = rem_b.denominator
                if c > b:
                    try_solution([a, b, c])
    
    # 4-term search
    for a in range(7, 15):
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        lower_b = max(a + 1, rem_a.denominator // rem_a.numerator + 1)
        upper_b = (3 * rem_a.denominator) // rem_a.numerator
        for b in range(lower_b, upper_b + 1):
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            lower_c = max(b + 1, rem_b.denominator // rem_b.numerator + 1)
            upper_c = (2 * rem_b.denominator) // rem_b.numerator
            if lower_c > upper_c:
                continue
            for c in range(lower_c, upper_c + 1):
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                if rem_c.numerator == 1:
                    d = rem_c.denominator
                    if d > c:
                        try_solution([a, b, c, d])
    
    if best_solution is not None:
        return best_solution
    
    return [8, 28, 1736]
