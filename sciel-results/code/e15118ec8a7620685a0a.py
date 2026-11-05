"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    # Try 2-term solutions: 1/a + 1/b = 5/31, a < b
    # 1/a + 1/b = (a+b)/(ab) = 5/31 => 31(a+b) = 5ab
    # => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31*31 = 961
    # So 5a-31 and 5b-31 are factor pairs of 961
    # 961 = 31*31 = 1*961
    # Factors: (1,961), (31,31), (961,1), (-1,-961), (-31,-31), (-961,-1)
    # 5a-31 = 1 => a = 32/5 not integer
    # 5a-31 = 31 => a = 62/5 not integer
    # No 2-term solutions with distinct positive integers
    
    # Try 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a, b: 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # => 1/c = (5a-31)/(31a) - 1/b = (b(5a-31) - 31a) / (31ab)
    # => c = 31ab / (b(5a-31) - 31a)
    # Need: b(5a-31) - 31a > 0, and c integer, c > b
    
    # a must satisfy: 1/a < 5/31 => a > 31/5 = 6.2, so a >= 7
    # Also 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18 for 3 terms
    # But we should be more careful: need 5/31 - 1/a > 0 => a > 31/5 = 6.2
    
    # For 3 terms with a < b < c:
    # 1/a + 1/b + 1/c = 5/31
    # Since a < b < c, we have 3/a > 5/31 => a < 93/5 = 18.6
    # And 1/a < 5/31 => a > 31/5 = 6.2
    # So a in [7, 18]
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # Now 1/b + 1/c = rem1, with a < b < c
        # 1/b < rem1 => b > 1/rem1
        # 2/b > rem1 => b < 2/rem1
        # Also b > a
        b_min = max(a + 1, (rem1.numerator + rem1.denominator - 1) // rem1.numerator + 1)
        # Actually: 1/b < rem1 => b > 1/rem1 => b >= floor(1/rem1) + 1
        # 1/b + 1/c < 2/b => rem1 < 2/b => b < 2/rem1
        # So b in range: b > 1/rem1 and b < 2/rem1
        # b_min = max(a+1, floor(1/rem1) + 1)
        # b_max = ceil(2/rem1) - 1
        
        one_over_rem1 = Fraction(1, 1) / rem1
        b_lower = max(a + 1, one_over_rem1.numerator // one_over_rem1.denominator + 1)
        two_over_rem1 = Fraction(2, 1) / rem1
        b_upper = two_over_rem1.numerator // two_over_rem1.denominator
        
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1/rem2
            if rem2.denominator == 1:
                c = rem2.numerator
                if c > b:
                    max_denom = c
                    if max_denom < best_max_denom:
                        best_max_denom = max_denom
                        best_solution = [a, b, c]
    
    # Try 4-term solutions if needed (only if no 3-term found with small max)
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # a in [7, ?]: 4/a > 5/31 => a < 124/5 = 24.8, so a <= 24
    # But we only need 4 terms if they give smaller max than 3-term solutions
    
    if best_max_denom > 100 or best_solution is None:
        for a in range(7, 25):
            if time.time() > deadline:
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            # 1/b + 1/c + 1/d = rem1, a < b < c < d
            # b > a, 3/b > rem1 => b < 3/rem1
            # 1/b < rem1 => b > 1/rem1
            one_over_rem1 = Fraction(1, 1) / rem1
            b_lower = max(a + 1, one_over_rem1.numerator // one_over_rem1.denominator + 1)
            three_over_rem1 = Fraction(3, 1) / rem1
            b_upper = three_over_rem1.numerator // three_over_rem1.denominator
            
            for b in range(b_lower, b_upper + 1):
                if time.time() > deadline:
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                # 1/c + 1/d = rem2, b < c < d
                # c > b, 2/c > rem2 => c < 2/rem2
                # 1/c < rem2 => c > 1/rem2
                one_over_rem2 = Fraction(1, 1) / rem2
                c_lower = max(b + 1, one_over_rem2.numerator // one_over_rem2.denominator + 1)
                two_over_rem2 = Fraction(2, 1) / rem2
                c_upper = two_over_rem2.numerator // two_over_rem2.denominator
                
                for c in range(c_lower, c_upper + 1):
                    if time.time() > deadline:
                        break
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    if rem3.denominator == 1:
                        d = rem3.numerator
                        if d > c:
                            max_denom = d
                            if max_denom < best_max_denom:
                                best_max_denom = max_denom
                                best_solution = [a, b, c, d]
    
    # Try 5-term solutions if needed
    if best_max_denom > 50 or best_solution is None:
        for a in range(7, 30):
            if time.time() > deadline:
                break
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            # 1/b + 1/c + 1/d + 1/e = rem1
            one_over_rem1 = Fraction(1, 1) / rem1
            b_lower = max(a + 1, one_over_rem1.numerator // one_over_rem1.denominator + 1)
            four_over_rem1 = Fraction(4, 1) / rem1
            b_upper = four_over_rem1.numerator // four_over_rem1.denominator
            
            for b in range(b_lower, b_upper + 1):
                if time.time() > deadline:
                    break
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                one_over_rem2 = Fraction(1, 1) / rem2
                c_lower = max(b + 1, one_over_rem2.numerator // one_over_rem2.denominator + 1)
                three_over_rem2 = Fraction(3, 1) / rem2
                c_upper = three_over_rem2.numerator // three_over_rem2.denominator
                
                for c in range(c_lower, c_upper + 1):
                    if time.time() > deadline:
                        break
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    one_over_rem3 = Fraction(1, 1) / rem3
                    d_lower = max(c + 1, one_over_rem3.numerator // one_over_rem3.denominator + 1)
                    two_over_rem3 = Fraction(2, 1) / rem3
                    d_upper = two_over_rem3.numerator // two_over_rem3.denominator
                    
                    for d in range(d_lower, d_upper + 1):
                        if time.time() > deadline:
                            break
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        if rem4.denominator == 1:
                            e = rem4.numerator
                            if e > d:
                                max_denom = e
                                if max_denom < best_max_denom:
                                    best_max_denom = max_denom
                                    best_solution = [a, b, c, d, e]
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy
    r, values = target, []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
