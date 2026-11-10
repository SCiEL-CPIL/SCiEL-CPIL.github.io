def construct():
    from fractions import Fraction
    import time
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_den = float('inf')
    
    start_time = time.time()
    timeout = 55  # seconds, leave buffer
    
    def search(remaining, min_den, current, max_den_so_far):
        nonlocal best_solution, best_max_den, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        if remaining == 0:
            if current:
                m = max(current)
                if m < best_max_den:
                    best_max_den = m
                    best_solution = list(current)
            return
        
        if remaining < 0:
            return
        
        # Pruning: if we've already exceeded best, skip
        if max_den_so_far >= best_max_den:
            return
        
        # Upper bound on next denominator
        # 1/d <= remaining => d >= 1/remaining
        # Also d >= min_den
        # And we need to be able to represent remaining - 1/d with denominators > d
        # Simple bound: d must satisfy 1/d < remaining (since we need more terms or it's exact)
        # Actually 1/d <= remaining, so d >= ceil(1/remaining)
        
        # Lower bound: d >= min_den
        # Upper bound: 1/d < remaining (strictly, since if 1/d == remaining we're done, 
        #              but if 1/d > remaining it's too big)
        # So d > 1/remaining, i.e., d >= floor(1/remaining) + 1 if 1/remaining is integer, else ceil(1/remaining)
        
        # Actually: we need remaining - 1/d >= 0, so 1/d <= remaining, so d >= 1/remaining
        # If remaining is a fraction a/b, then 1/d <= a/b => b <= a*d => d >= b/a
        # So d >= ceil(b/a)
        
        num, den = remaining.numerator, remaining.denominator
        # d >= ceil(den/num)
        lower = (den + num - 1) // num
        if lower < min_den:
            lower = min_den
        
        # Upper bound: we need at least one more term after this (unless remaining == 1/d)
        # If we use d, remaining - 1/d must be representable. 
        # Simple upper bound: d such that 1/d < remaining (to leave positive remainder)
        # 1/d < num/den => den < num*d => d > den/num
        # So d >= floor(den/num) + 1 = lower (same as above)
        # For an upper bound, note that if we have k more terms each at least 1/(d+1), 
        # then remaining - 1/d <= k/(d+1) for some k. This is complex.
        # Simple heuristic upper bound: try up to some limit or best_max_den - 1
        upper = best_max_den - 1  # must be strictly less than current best max
        if upper < lower:
            return
        
        # Try denominators from lower to upper
        for d in range(lower, upper + 1):
            if time.time() - start_time > timeout:
                return
            
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break  # d too small... wait, as d increases, 1/d decreases, so new_remaining increases
            # Actually as d increases, 1/d decreases, so new_remaining = remaining - 1/d increases
            # So if new_remaining < 0, we need smaller d. But we're going up, so this won't happen if lower is correct.
            
            new_max = max(max_den_so_far, d)
            if new_max >= best_max_den:
                continue
            
            current.append(d)
            search(new_remaining, d + 1, current, new_max)
            current.pop()
    
    search(target, 1, [], 0)
    
    if best_solution is None:
        # Fallback to the original greedy
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_solution
