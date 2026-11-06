def construct():
    import time
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    # Known candidate: [8, 31, 248] with max=248
    candidate = [8, 31, 248]
    # Verify
    assert Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248) == target, "Candidate invalid"
    
    best_max = 248
    best_solution = list(candidate)
    
    # Try to find a solution with max < 248
    # 2-term: no solution exists (verified analytically)
    # 3-term: check if any 3-term solution has all terms < 248
    # a < b < c < 248, and 1/a + 1/b + 1/c = 5/31
    # a >= 7 (since 3/6 > 5/31)
    # For each a, b, check if c exists and c < 248
    
    for a in range(7, 248):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # b > a, and 1/b < r1 => b > 1/r1
        # Also 2/b > r1 => b < 2/r1
        b_min = a + 1
        b_max = min(int(2 / r1) + 1, 247)
        if b_max < b_min:
            continue
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # c = 1/r2 must be integer and c > b and c < 248
            if r2.denominator == 1:
                c = r2.denominator  # This is wrong
            # r2 = n2/d2, c = d2/n2 must be integer
            n2 = r2.numerator
            d2 = r2.denominator
            if n2 == 0:
                continue
            if d2 % n2 != 0:
                continue
            c = d2 // n2
            if c <= b or c >= 248:
                continue
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                m = max(a, b, c)
                if m < best_max:
                    best_max = m
                    best_solution = [a, b, c]
    
    # 4-term: check if any 4-term solution has all terms < 248
    # a < b < c < d < 248
    # a >= 7, and 4/a > 5/31 => a < 124/5 = 24.8
    for a in range(7, 25):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            continue
        # b > a, 3/b > r1 => b < 3/r1
        b_max = min(int(3 / r1) + 1, 247)
        for b in range(a + 1, b_max + 1):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                continue
            # c > b, 2/c > r2 => c < 2/r2
            c_max = min(int(2 / r2) + 1, 247)
            for c in range(b + 1, c_max + 1):
                if time.time() > deadline:
                    break
                r3 = r2 - Fraction(1, c)
                if r3 <= 0:
                    continue
                n3 = r3.numerator
                d3 = r3.denominator
                if n3 == 0:
                    continue
                if d3 % n3 != 0:
                    continue
                d = d3 // n3
                if d <= c or d >= 248:
                    continue
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                    m = max(a, b, c, d)
                    if m < best_max:
                        best_max = m
                        best_solution = [a, b, c, d]
    
    return best_solution
