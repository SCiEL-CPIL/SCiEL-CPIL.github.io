def construct():
    from fractions import Fraction
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_values = None
    
    def try_complete_greedy(r, used, values, max_terms=256, search_width=200):
        """Greedy completion with small search width."""
        if r == 0:
            return values[:]
        if len(values) >= max_terms:
            return None
        
        d_min = (r.denominator + r.numerator - 1) // r.numerator
        for d in range(d_min, d_min + search_width):
            if d > 10000000 or d in used:
                continue
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            used.add(d)
            result = try_complete_greedy(new_r, used, values + [d], max_terms, search_width)
            used.discard(d)
            if result is not None:
                return result
        return None
    
    # Search over first two denominators
    for d1 in range(7, 150):
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        
        d2_min = (rem1.denominator + rem1.numerator - 1) // rem1.numerator
        for d2 in range(d2_min, d2_min + 150):
            if d2 == d1 or d2 < 1:
                continue
            rem2 = rem1 - Fraction(1, d2)
            if rem2 < 0:
                break
            if rem2 == 0:
                values = [d1, d2]
                if max(values) < best_max:
                    best_max = max(values)
                    best_values = values[:]
                continue
            
            used = {d1, d2}
            result = try_complete_greedy(rem2, used, [d1, d2])
            if result is not None:
                max_d = max(result)
                if max_d < best_max:
                    best_max = max_d
                    best_values = result[:]
    
    if best_values:
        return best_values
    else:
        # Fallback
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
