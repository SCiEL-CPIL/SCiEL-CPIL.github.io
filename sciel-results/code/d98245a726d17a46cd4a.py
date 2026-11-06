def construct():
    import time
    from fractions import Fraction
    
    target = Fraction(5, 31)
    deadline = time.time() + 85
    
    # Known candidate: [8, 31, 248] with max=248
    best_solution = [8, 31, 248]
    best_max = 248
    
    def try_update(denoms):
        nonlocal best_max, best_solution
        m = max(denoms)
        if m < best_max:
            s = sum(Fraction(1, d) for d in denoms)
            if s == target:
                best_max = m
                best_solution = list(denoms)
                return True
        return False
    
    # Search for solutions with all denominators <= 247
    # Use backtracking with pruning
    
    def search(remaining_terms, remaining_val, min_denom, current):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        
        if remaining_terms == 0:
            return remaining_val == 0
        
        if remaining_terms == 1:
            n, d = remaining_val.numerator, remaining_val.denominator
            if n == 0:
                return False
            if d % n != 0:
                return False
            denom = d // n
            if denom > min_denom and denom <= 247:
                if try_update(current + [denom]):
                    return True
            return False
        
        # Pruning: check if it's possible to achieve remaining_val with remaining_terms
        # distinct denominators all > min_denom and <= 247
        
        # Maximum possible sum: use smallest available denominators
        if min_denom + remaining_terms > 247:
            return False
        max_sum = sum(Fraction(1, min_denom + i) for i in range(1, remaining_terms + 1))
        if max_sum < remaining_val:
            return False
        
        # Minimum possible sum: use largest available denominators
        min_sum = sum(Fraction(1, 248 - i) for i in range(1, remaining_terms + 1))
        if min_sum > remaining_val:
            return False
        
        # Next denominator a:
        # 1/a < remaining_val => a > 1/remaining_val
        # remaining_terms/a > remaining_val => a < remaining_terms/remaining_val
        a_min = max(min_denom + 1, int(1 / remaining_val) + 1)
        a_max = min(int(remaining_terms / remaining_val), 247)
        
        if a_min > a_max:
            return False
        
        for a in range(a_min, a_max + 1):
            new_remaining = remaining_val - Fraction(1, a)
            if new_remaining < 0:
                break
            if search(remaining_terms - 1, new_remaining, a, current + [a]):
                return True
        return False
    
    # Try 3 terms
    search(3, target, 6, [])
    if best_max < 248:
        return best_solution
    
    # Try 4 terms
    search(4, target, 6, [])
    if best_max < 248:
        return best_solution
    
    # Try 5 terms
    search(5, target, 6, [])
    if best_max < 248:
        return best_solution
    
    # Try 6 terms
    search(6, target, 6, [])
    
    return best_solution
