def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    
    # Known solution with max denominator 1736
    # 5/31 = 1/8 + 1/28 + 1/1736
    best_vals = [8, 28, 1736]
    best_max = 1736
    
    # Search for a solution with max denominator < best_max
    # We check expansions of length 3 and 4.
    # Length 3: 1/d1 + 1/d2 + 1/d3 = 5/31
    # Length 4: 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    
    for d1 in range(7, 100):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        for d2 in range(d1 + 1, 1000):
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            
            # Check if r2 can be a single unit fraction 1/d3
            # d3 = 1/r2
            if r2.numerator == 1:
                d3 = r2.denominator
                if d3 > d2 and d3 < best_max:
                    return [d1, d2, d3]
            
            # Check if r2 can be sum of two unit fractions 1/d3 + 1/d4
            # We require max(d3, d4) < best_max
            # If 2/r2 > best_max, then min possible max(d3,d4) is > best_max
            # Actually, min max(d3,d4) is approx 2/r2.
            # So if 2/r2 >= best_max, skip.
            if r2 < Fraction(2, best_max):
                continue
            
            # Iterate d4 from d2+1 to best_max-1
            # d3 = d4 / (r2*d4 - 1)
            for d4 in range(d2 + 1, best_max):
                denom = r2 * d4 - 1
                if denom <= 0:
                    continue
                num = d4
                if num % denom == 0:
                    d3 = num // denom
                    if d3 > d2 and d3 < d4: # d3 < d4 to avoid duplicate checks, and distinct
                        # Ensure d3 is not in [d1, d2, d4]
                        if d3 != d1 and d3 != d2 and d3 != d4:
                            return [d1, d2, d3, d4]
                    elif d3 > d2 and d3 > d4:
                        # This case is covered by swapping d3, d4 in logic? 
                        # No, we iterate d4. If d3 > d4, then the pair (d4, d3) is valid.
                        # But we assume d3 < d4 in the loop logic? 
                        # Let's just check if d3 is integer and distinct.
                        if d3 != d1 and d3 != d2 and d3 != d4:
                            return [d1, d2, d3, d4]

    # If no better solution found, return the known one
    return best_vals
