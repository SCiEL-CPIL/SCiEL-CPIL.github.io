import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 80
    best_solution = None
    best_max_den = float('inf')
    
    def solve_two_terms(r):
        """
        Solve r = 1/a + 1/b for distinct integers a, b.
        Returns (a, b) with a < b, or None.
        """
        if r <= 0:
            return None
        p = r.numerator
        q = r.denominator
        
        # (p*a - q)(p*b - q) = q^2
        N = q * q
        # Iterate divisors of N up to sqrt(N)
        limit = int(N ** 0.5) + 1
        
        # Optimization: We only care about solutions where a, b are not too large?
        # No, we just find all valid pairs and let the caller filter.
        # However, N can be large.
        # For 5/31, d1=7 -> r1=4/217 -> q=217 -> N=47089. sqrt ~ 217.
        # d1=7, d2=55 -> r2 = 4/217 - 1/55 = (220-217)/11935 = 3/11935. q=11935. N ~ 1.4e8. sqrt ~ 12000.
        # This is fast enough.
        
        for x in range(1, limit + 1):
            if N % x != 0:
                continue
            y = N // x
            # Check if x and y are valid factors
            # a = (x + q) / p
            # b = (y + q) / p
            if (x + q) % p != 0:
                continue
            if (y + q) % p != 0:
                continue
            
            a = (x + q) // p
            b = (y + q) // p
            
            if a == b:
                continue
            
            if a < 1 or b < 1:
                continue
            
            # Ensure a < b
            if a > b:
                a, b = b, a
                
            return (a, b)
            
        return None

    def update_best(sol):
        nonlocal best_solution, best_max_den
        if not sol:
            return
        if len(sol) > 256:
            return
        # Check distinctness
        if len(set(sol)) != len(sol):
            return
        # Check range
        for d in sol:
            if d < 1 or d > 10000000:
                return
        # Check sum
        s = sum(Fraction(1, d) for d in sol)
        if s != target:
            return
        m = max(sol)
        if m < best_max_den:
            best_max_den = m
            best_solution = sol[:]

    # Search for 3-term solutions: 1/d1 + 1/d2 + 1/d3 = 5/31
    for d1 in range(7, 20):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        # Try to complete with 2 terms
        pair = solve_two_terms(r1)
        if pair:
            d2, d3 = pair
            if d1 < d2 < d3:
                update_best([d1, d2, d3])

    # Search for 4-term solutions: 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    for d1 in range(7, 15):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        
        # d2 must be > 1/r1 to keep r2 > 0? No, 1/d2 < r1 is required for r2 > 0?
        # If 1/d2 >= r1, then r2 <= 0, and we can't add positive terms 1/d3 + 1/d4.
        # So we need 1/d2 < r1 => d2 > 1/r1.
        min_d2 = int(1 / r1) + 1
        if min_d2 <= d1:
            min_d2 = d1 + 1
            
        # Upper bound for d2:
        # We want max(d1..d4) to be small.
        # If d2 is very large, d3, d4 will be very large.
        # Let's iterate d2 up to some limit, say 100 or best_max_den.
        max_d2 = 100
        if best_max_den < float('inf'):
            max_d2 = min(max_d2, best_max_den)
            
        for d2 in range(min_d2, max_d2 + 1):
            if time.time() > deadline:
                break
            if d2 <= d1:
                continue
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
                
            # Try to complete with 2 terms
            pair = solve_two_terms(r2)
            if pair:
                d3, d4 = pair
                if d2 < d3 < d4:
                    # Check distinctness with d1, d2
                    if d1 not in (d3, d4) and d2 not in (d3, d4):
                        update_best([d1, d2, d3, d4])

    # Fallback: Greedy if no small solution found (unlikely)
    if best_solution is None:
        r = target
        sol = []
        used = set()
        min_d = 1
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            sol.append(d)
            used.add(d)
            r -= Fraction(1, d)
            min_d = d + 1
        best_solution = sol
        best_max_den = max(sol)

    return best_solution
