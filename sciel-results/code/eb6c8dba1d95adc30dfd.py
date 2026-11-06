def construct():
    from fractions import Fraction
    import time
    
    target = Fraction(5, 31)
    best_solution = None
    best_max = 10**7
    
    start_time = time.time()
    deadline = start_time + 45
    
    def backtrack(k, remaining, min_d, current):
        nonlocal best_solution, best_max, deadline
        
        if time.time() > deadline:
            return
        
        if k == 0:
            if remaining == 0:
                max_d = max(current)
                if max_d < best_max:
                    best_max = max_d
                    best_solution = list(current)
            return
        
        if k == 1:
            # 1/d = remaining
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d:
                    current.append(d)
                    max_d = max(current)
                    if max_d < best_max:
                        best_max = max_d
                        best_solution = list(current)
                    current.pop()
            return
        
        r_float = float(remaining)
        # 1/d < remaining => d > 1/remaining
        low = max(min_d, int(1 / r_float) + 1)
        # (k)/(d) > remaining => d < k/remaining
        high = int(k / r_float)
        
        if low > high:
            return
        
        if current and max(current) > best_max:
            return

        for d in range(low, high + 1):
            rem = remaining - Fraction(1, d)
            if rem < 0:
                continue
            # Prune: max sum of k-1 terms > d is (k-1)/(d+1)
            if rem > Fraction(k - 1, d + 1):
                continue
            
            current.append(d)
            backtrack(k - 1, rem, d + 1, current)
            current.pop()

    for k in range(3, 11):
        if time.time() > deadline:
            break
        backtrack(k, target, 1, [])

    if best_solution:
        return best_solution
    else:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
