def construct():
    from fractions import Fraction
    import time

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    target = Fraction(5, 31)
    
    # Global best solution
    best_solution = None
    best_max_den = float('inf')

    # Helper to update best solution
    def update_best(denoms, max_d):
        nonlocal best_solution, best_max_den
        if max_d < best_max_den:
            best_max_den = max_d
            best_solution = denoms[:]

    # Quick 3-term search to set an initial bound
    for a in range(7, 19):
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        max_b = int(2 / rem1)
        for b in range(a + 1, max_b + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            if rem2.denominator % rem2.numerator == 0:
                c = rem2.denominator // rem2.numerator
                if c > b:
                    if c < best_max_den:
                        best_max_den = c
                        best_solution = [a, b, c]
                    break
        if best_solution is not None:
            break

    if best_solution is None:
        best_solution = [8, 28, 1736]
        best_max_den = 1736

    def dfs(remaining_sum, last_denom, terms_left, current_denoms):
        nonlocal best_max_den, best_solution
        
        if time.time() > deadline - 2.0:
            return
        
        if terms_left == 1:
            if remaining_sum.numerator != 1:
                return
            d = remaining_sum.denominator
            if d <= last_denom:
                return
            if d >= best_max_den:
                return
            update_best(current_denoms + [d], d)
            return
        
        min_d = last_denom + 1
        max_d = int(terms_left / remaining_sum)
        
        if max_d >= best_max_den:
            max_d = best_max_den - 1
            
        if min_d > max_d:
            return
            
        for d in range(min_d, max_d + 1):
            rem = remaining_sum - Fraction(1, d)
            if rem <= 0:
                continue
                
            m = terms_left - 1
            
            # Prune: max possible sum with m terms > d
            max_possible_sum = 0
            for i in range(1, m + 1):
                max_possible_sum += Fraction(1, d + i)
            if rem > max_possible_sum:
                continue
            
            # Prune: min possible sum with m terms < best_max_den
            if rem < Fraction(m, best_max_den - 1):
                continue

            current_denoms.append(d)
            dfs(rem, d, m, current_denoms)
            current_denoms.pop()

    for k in range(3, 11):
        if time.time() > deadline - 5.0:
            break
        current_denoms = []
        dfs(target, 0, k, current_denoms)

    if best_solution is None:
        return [8, 28, 1736]
        
    return best_solution
