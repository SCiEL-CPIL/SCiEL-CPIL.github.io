def construct():
    from fractions import Fraction
    import time
    from math import gcd
    from collections import defaultdict

    deadline = time.time() + 85
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max = 10000001

    def update_best(sol):
        nonlocal best_solution, best_max
        if sol:
            mx = max(sol)
            if mx < best_max:
                best_max = mx
                best_solution = sol

    # Meet in the middle for 3 terms:
    # 1/a + 1/b + 1/c = 5/31
    # Precompute pairs (b, c) with b < c
    # Store in dict: sum -> (b, c)
    # Then iterate a, check if target - 1/a is in dict.
    
    # To minimize max denominator, we should search in increasing order of max denominator.
    # Or just search a range and keep the best.
    # Since we want to minimize the MAX, we can bound the search.
    # Let's try to find solutions with max denominator < 1000 first.
    
    def find_3_term(limit):
        # limit is the max denominator allowed
        pair_sums = {}
        # b, c must be >= 7
        # 1/b + 1/c < 5/31
        # b < c <= limit
        for b in range(7, limit + 1):
            for c in range(b + 1, limit + 1):
                s = Fraction(1, b) + Fraction(1, c)
                if s < target:
                    # Store the pair with the smallest max (which is c)
                    # If multiple pairs have same sum, keep the one with smallest c?
                    # Actually, we just need any pair.
                    # But we want to minimize the global max.
                    # If we fix 'a', the max is max(a, b, c).
                    # We will check this later.
                    if s not in pair_sums:
                        pair_sums[s] = (b, c)
                    else:
                        # Keep the pair with smaller max (c)
                        if c < pair_sums[s][1]:
                            pair_sums[s] = (b, c)
        
        for a in range(7, limit + 1):
            rem = target - Fraction(1, a)
            if rem in pair_sums:
                b, c = pair_sums[rem]
                if b > a and c > b:
                    sol = [a, b, c]
                    update_best(sol)

    # Meet in the middle for 4 terms:
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # Precompute pairs (c, d) with c < d
    # Iterate a, b. Check if target - 1/a - 1/b is in pair_sums.
    
    def find_4_term(limit):
        pair_sums = {}
        for c in range(7, limit + 1):
            for d in range(c + 1, limit + 1):
                s = Fraction(1, c) + Fraction(1, d)
                if s < target:
                    if s not in pair_sums:
                        pair_sums[s] = (c, d)
                    else:
                        if d < pair_sums[s][1]:
                            pair_sums[s] = (c, d)
        
        for a in range(7, limit + 1):
            if time.time() > deadline: return
            for b in range(a + 1, limit + 1):
                rem = target - Fraction(1, a) - Fraction(1, b)
                if rem in pair_sums:
                    c, d = pair_sums[rem]
                    if c > b and d > c:
                        sol = [a, b, c, d]
                        update_best(sol)

    # DFS for 5 terms (if needed)
    def dfs_5_term(limit):
        # This is expensive, only run if no 3 or 4 term solution found with small max
        # Or run with tighter limit
        pass

    # Strategy:
    # 1. Try 3-term with limit 500
    # 2. Try 4-term with limit 200 (since 4 terms can have smaller max than 3? 
    #    Actually, more terms usually allow smaller max denominators because you can use smaller numbers.
    #    e.g. 1/7 + 1/8 + 1/9 + 1/10 vs 1/7 + 1/100 + 1/1000.
    #    So we should prioritize searching for solutions with smaller limits, regardless of term count.
    
    # Let's search for any solution with max denominator <= L, for L = 10, 20, 50, 100, 200, 500.
    # For each L, check 3-term and 4-term.
    
    limits = [10, 20, 50, 100, 200, 500, 1000]
    
    for L in limits:
        if time.time() > deadline: break
        
        # 3-term search with max denom L
        # We can optimize by only generating pairs with c <= L
        # And a <= L
        # If we find a solution, update best.
        # We don't need to continue searching for this L if we found one?
        # No, we want the MIN max. So if we find one with max <= L, we record it.
        # We should then try to find one with smaller max?
        # The loop goes increasing L. So the first found is the smallest max?
        # Not necessarily. A 4-term solution might have max 15, while 3-term has min max 20.
        # So we should check all term counts for each L.
        
        found_at_L = False
        
        # 3-term
        pair_sums = {}
        for b in range(7, L + 1):
            for c in range(b + 1, L + 1):
                s = Fraction(1, b) + Fraction(1, c)
                if s < target:
                    if s not in pair_sums:
                        pair_sums[s] = (b, c)
                    elif c < pair_sums[s][1]:
                        pair_sums[s] = (b, c)
        
        for a in range(7, L + 1):
            rem = target - Fraction(1, a)
            if rem in pair_sums:
                b, c = pair_sums[rem]
                if b > a and c > b:
                    sol = [a, b, c]
                    mx = max(sol)
                    if mx < best_max:
                        best_max = mx
                        best_solution = sol
                        found_at_L = True
                        # We can break if mx == L? No, maybe another combo has smaller mx?
                        # But mx <= L. And we are iterating L upwards.
                        # So the first L that yields a solution gives the minimal possible max for that term count?
                        # Not strictly, because a solution with max 15 might be found at L=20.
                        # But we iterate L=10, 20...
                        # If we find a solution at L=20 with max 15, it would have been found at L=15?
                        # We skipped 15.
                        # So we should iterate L more densely or just rely on the fact that we update best_max.
        
        # 4-term
        pair_sums = {}
        for c in range(7, L + 1):
            for d in range(c + 1, L + 1):
                s = Fraction(1, c) + Fraction(1, d)
                if s < target:
                    if s not in pair_sums:
                        pair_sums[s] = (c, d)
                    elif d < pair_sums[s][1]:
                        pair_sums[s] = (c, d)
        
        for a in range(7, L + 1):
            if time.time() > deadline: break
            for b in range(a + 1, L + 1):
                rem = target - Fraction(1, a) - Fraction(1, b)
                if rem in pair_sums:
                    c, d = pair_sums[rem]
                    if c > b and d > c:
                        sol = [a, b, c, d]
                        mx = max(sol)
                        if mx < best_max:
                            best_max = mx
                            best_solution = sol
                            found_at_L = True

    if best_solution:
        return best_solution

    # Fallback
    return [7, 8, 9, 10, 11] # Invalid but prevents crash
