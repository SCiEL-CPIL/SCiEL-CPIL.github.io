def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    target = Fraction(5, 31)
    
    best_solution = None
    best_max_den = float('inf')
    
    def search(remainder, min_den, current_dens, max_den_allowed, deadline):
        nonlocal best_solution, best_max_den
        if time.time() > deadline:
            return
        if not remainder:
            if current_dens:
                mx = max(current_dens)
                if mx < best_max_den:
                    best_max_den = mx
                    best_solution = list(current_dens)
            return
        if len(current_dens) >= 256:
            return
        # The next denominator d must satisfy:
        # 1. d >= min_den
        # 2. d > remainder.denominator / remainder.numerator (i.e., 1/d < remainder)
        # 3. d <= max_den_allowed
        # 4. The remaining (terms-1) fractions with denominators > d must sum to remainder - 1/d
        #    Maximum possible sum with k terms all > d is k/(d+1) (roughly, but more precisely bounded)
        
        p, q = remainder.numerator, remainder.denominator
        # d must be >= ceil(q/p) to have 1/d <= remainder
        d_min = max(min_den, (q + p - 1) // p)
        d_max = max_den_allowed
        
        if d_min > d_max:
            return
        
        # Bound: with remaining terms, the max sum of reciprocals with denominators > d
        # is at most (256 - len(current_dens)) / (d + 1) for a loose bound
        # But tighter: the remainder after choosing d is r' = remainder - 1/d
        # We need r' to be expressible as sum of at most (256 - len(current_dens) - 1) distinct unit fractions > d
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return
            if 1 / d >= remainder:
                continue
            new_rem = remainder - Fraction(1, d)
            if new_rem < 0:
                break
            # Check if new_rem can be completed: 
            # The minimum denominator for next term is d+1
            # Maximum sum with n terms all >= d+1 is n/(d+1)
            remaining_terms = 255 - len(current_dens)
            if remaining_terms <= 0:
                if new_rem == 0:
                    if d < best_max_den:
                        best_max_den = d
                        best_solution = current_dens + [d]
                continue
            # Loose upper bound on sum of remaining terms
            if new_rem > remaining_terms / (d + 1):
                continue
            # Also, new_rem must be positive
            current_dens.append(d)
            search(new_rem, d + 1, current_dens, max_den_allowed, deadline)
            current_dens.pop()
    
    # Try increasing max denominator bounds
    # Start with a reasonable bound and increase
    deadline = start_time + 50  # 50 second budget
    
    # First, try to find any solution with small max denominator
    # The theoretical minimum: 5/31 < 1/6, so we need at least 2 terms
    # With 2 terms: 1/a + 1/b = 5/31, a < b
    # 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 961
    # 961 = 31^2, so factors: 1*961, 31*31
    # 5a-31=1 => a=32/5 not integer
    # 5a-31=31 => a=12.4 not integer
    # 5a-31=961 => a=198.4 not integer
    # So no 2-term solution with distinct denominators
    
    # Try 3 terms: search with max_den from small to large
    # Based on observation, answer is around 248
    
    for max_den in [200, 250, 300, 400, 500, 800, 1000, 2000, 5000, 10000]:
        if time.time() > deadline:
            break
        best_solution = None
        best_max_den = float('inf')
        search(target, 1, [], max_den, deadline)
        if best_solution is not None:
            return best_solution
    
    # If we found something, return it
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
