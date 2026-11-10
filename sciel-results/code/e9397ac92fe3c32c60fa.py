"""Rewritten program: exhaustive bounded search over first two denominators, exact completion for the third."""
# EVOLVE-BLOCK-START
def construct():
    # Target: 1/d1 + 1/d2 + 1/d3 = 5/31 with distinct positive integers d1 < d2 < d3.
    # 5/31 ≈ 0.16129.
    # For fixed d1, d2, we need 1/d3 = 5/31 - 1/d1 - 1/d2 = r.
    # r must be positive, and d3 = 1/r must be an integer > d2.
    #
    # We enumerate d1 from 8 upward (since 1/7 + 1/8 + 1/9 > 5/31, d1 must be >= 8 for any 3-term
    # solution with d1 < d2 < d3 and distinct denominators; actually d1 could be larger, but we bound d3).
    # We want to minimize d3, the largest denominator.
    #
    # From the seed analysis: [8, 31, 248] is a solution with max 248.
    # We will search for any solution with max < 248 to see if we can improve.
    # If none found, return [8, 31, 248].
    
    best = None
    best_max = 10**9
    
    # d1 must be at least ceil(31/5) = 7, but 1/7 alone is too large if we need 3 terms? 
    # Actually 1/7 ≈ 0.1428, 5/31 ≈ 0.1613. So 1/7 < 5/31. 
    # But if d1=7, then 1/d2 + 1/d3 = 5/31 - 1/7 = (35-31)/217 = 4/217.
    # Then d2 > 217/4 ≈ 54.25. So d2 >= 55.
    # Then 1/d3 = 4/217 - 1/d2. For d2=55, 1/d3 = 4/217 - 1/55 = (220-217)/(217*55) = 3/11935. d3 not integer.
    # Let's just iterate d1 from 7 to 30.
    
    for d1 in range(7, 31):
        # Remainder after d1
        # r1 = 5/31 - 1/d1 = (5*d1 - 31) / (31*d1)
        num1 = 5 * d1 - 31
        den1 = 31 * d1
        if num1 <= 0:
            continue
        # We need 1/d2 + 1/d3 = num1/den1
        # d2 > d1
        # Also 1/d2 < num1/den1 => d2 > den1/num1
        # And 2/d2 > num1/den1 (since d3 > d2 => 1/d3 < 1/d2 => 1/d2 + 1/d3 < 2/d2)
        # So d2 < 2*den1/num1
        low_d2 = den1 // num1 + 1
        high_d2 = (2 * den1) // num1 + 1
        if low_d2 <= d1:
            low_d2 = d1 + 1
        if low_d2 >= high_d2:
            continue
            
        for d2 in range(low_d2, high_d2):
            # 1/d3 = num1/den1 - 1/d2 = (num1*d2 - den1) / (den1*d2)
            num3 = num1 * d2 - den1
            den3 = den1 * d2
            if num3 <= 0:
                break
            if den3 % num3 != 0:
                continue
            d3 = den3 // num3
            if d3 <= d2:
                continue
            # Found a solution [d1, d2, d3]
            if d3 < best_max:
                best_max = d3
                best = [d1, d2, d3]
                
    if best:
        return best
    else:
        # Fallback to the known solution
        return [8, 31, 248]
# EVOLVE-BLOCK-END
