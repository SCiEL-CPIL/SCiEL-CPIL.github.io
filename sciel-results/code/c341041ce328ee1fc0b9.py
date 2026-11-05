def construct():
    from fractions import Fraction
    import time
    
    start = time.time()
    deadline = start + 85.0
    target = Fraction(5, 31)
    
    best_max = float('inf')
    best_sol = None
    
    def search(remaining, current_terms, min_den, max_allowed, terms_remaining):
        if time.time() > deadline - 3:
            return None
        if remaining == 0:
            return list(current_terms)
        if terms_remaining <= 0:
            return None
        if remaining < 0:
            return None
        
        lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        lower = max(lower, min_den)
        
        upper = (terms_remaining * remaining.denominator) // remaining.numerator
        upper = min(upper, max_allowed)
        
        if lower > upper:
            return None
            
        for d in range(lower, upper + 1):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if terms_remaining >= 1:
                    return list(current_terms) + [d]
            elif terms_remaining > 1:
                result = search(new_rem, current_terms + [d], d + 1, max_allowed, terms_remaining - 1)
                if result is not None:
                    return result
        return None
    
    for max_allowed in range(7, 200):
        if time.time() > deadline - 5:
            break
        found = False
        for first in range(7, max_allowed + 1):
            rem = target - Fraction(1, first)
            if rem < 0:
                break
            if rem == 0:
                best_max = max_allowed
                best_sol = [first]
                found = True
                break
            result = search(rem, [first], first + 1, max_allowed, 255)
            if result is not None:
                best_max = max_allowed
                best_sol = result
                found = True
                break
        if found:
            break
    
    if best_sol is not None:
        return best_sol
    
    r = target
    values = []
    while r and len(values) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
