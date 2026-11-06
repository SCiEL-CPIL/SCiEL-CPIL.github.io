def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 75
    
    target = Fraction(5, 31)
    best_max = 10000000
    best_solution = None
    
    def greedy_complete(remaining, min_d, max_allowed):
        sol = []
        r = remaining
        md = min_d
        while r > 0:
            if time.time() > deadline:
                return None
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < md:
                d = md
            if d > max_allowed or d > 10000000:
                return None
            sol.append(d)
            r -= Fraction(1, d)
            if r < 0:
                return None
            md = d + 1
            if len(sol) > 250:
                return None
        return sol
    
    for d1 in range(1, 30):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, 100):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            for d3 in range(d2 + 1, 200):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, d3)
                if r3 <= 0:
                    continue
                sol = greedy_complete(r3, d3 + 1, best_max - 1)
                if sol is not None:
                    m = max(d1, d2, d3)
                    if m < best_max:
                        best_max = m
                        best_solution = [d1, d2, d3] + sol
    
    if best_solution is None:
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
