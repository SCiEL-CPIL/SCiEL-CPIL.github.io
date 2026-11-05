def construct():
    import time
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    from fractions import Fraction
    from math import isqrt, gcd

    target = Fraction(5, 31)

    best_max_den = 10000001
    best_solution = None

    # Phase 1: Exhaustive 3-term search
    # 1/a + 1/b + 1/c = 5/31, WLOG a <= b <= c
    # a >= 7 (since 1/a < 5/31), and a <= 18 (since 3/a >= 5/31)
    # But to be safe, search a up to best_max_den
    for a in range(7, 200):
        if time.time() - start_time > DEADLINE:
            break
        if a >= best_max_den:
            break

        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a

        g = gcd(num, den)
        n = num // g
        d = den // g

        d2 = d * d
        limit = isqrt(d2)

        for u in range(1, limit + 1):
            if d2 % u != 0:
                continue
            v = d2 // u

            bu = u + d
            bv = v + d

            if bu % n == 0 and bv % n == 0:
                b = bu // n
                c = bv // n

                if b <= 0 or c <= 0:
                    continue
                if b == a or c == a or b == c:
                    continue
                if b > 10000000 or c > 10000000:
                    continue

                m = max(a, b, c)
                if m < best_max_den:
                    check = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                    if check == target:
                        best_max_den = m
                        best_solution = [a, b, c]

    if best_solution is not None:
        return best_solution

    # Phase 2: DFS for 4-term solutions with max denominator < some bound
    # Try to find 1/a + 1/b + 1/c + 1/d = 5/31 with max(a,b,c,d) < best bound
    # We'll try max_den_bound from small values upward
    # But first, let's just do a bounded DFS

    def dfs4(remaining, min_den, depth, max_bound, current):
        """Find denominators for remaining fraction, each >= min_den, <= max_bound, distinct from current."""
        nonlocal best_max_den, best_solution
        if time.time() - start_time > DEADLINE:
            return False
        if remaining == 0:
            if depth >= 4:
                m = max(current)
                if m < best_max_den:
                    best_max_den = m
                    best_solution = list(current)
            return True
        if remaining < 0:
            return False
        if depth == 4:
            return False

        # remaining = n/d
        n = remaining.numerator
        d = remaining.denominator

        # Next denominator x must satisfy:
        # 1/x < remaining => x > d/n
        # Also x >= min_den, x <= max_bound
        # And we need remaining - 1/x > 0 or == 0
        # If depth remaining terms = k, then k/x >= remaining => x <= k*d/n (roughly)
        # More precisely: remaining - 1/x must be >= 0 and representable by (depth - len(current)) terms each >= x
        
        k = 4 - len(current)  # number of terms still to place
        # x must be > d/n (so 1/x < remaining)
        # x must be <= k * d / n (so that k/x >= remaining, i.e., we can still reach remaining)
        # Actually: remaining = sum of k terms each >= 1/x (since next terms >= x)
        # So remaining <= k/x => x <= k/remaining = k*d/n
        # And remaining > 1/x (strictly, since other terms positive) => x > d/n
        
        lo = max(min_den, d // n + 1)
        hi = min(max_bound, (k * d) // n)
        
        if lo > hi:
            return False

        found = False
        for x in range(lo, hi + 1):
            if time.time() - start_time > DEADLINE:
                return False
            if x in current:
                continue
            new_rem = remaining - Fraction(1, x)
            if new_rem < 0:
                break
            if new_rem == 0:
                if len(current) + 1 == 4:
                    m = max(current + [x])
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = list(current) + [x]
                    found = True
                continue
            current.append(x)
            if dfs4(new_rem, x + 1, depth + 1, max_bound, current):
                found = True
            current.pop()
            if found and best_solution is not None:
                # We found something, but keep searching for better
                pass
        return found

    # Try 4-term with progressively larger max bounds
    # Start from a reasonable lower bound
    # For 4 terms summing to 5/31, min possible max is at least ceil(31/5*4/4)... 
    # Actually min max >= 8 (since 4/8 = 1/2 > 5/31, need at least 4 terms each <= 1/8)
    # Let's try max bounds from 20 upward
    
    for max_bound in range(20, 200):
        if time.time() - start_time > DEADLINE:
            break
        if max_bound >= best_max_den:
            break
        dfs4(target, 7, 0, max_bound, [])
        if best_solution is not None and best_max_den <= max_bound:
            break

    if best_solution is not None:
        return best_solution

    # Fallback: greedy
    r = Fraction(5, 31)
    values = []
    while r > 0:
        d_val = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d_val)
        r -= Fraction(1, d_val)
    return values
