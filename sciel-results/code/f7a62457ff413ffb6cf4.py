"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    
    target = Fraction(5, 31)
    
    # For each candidate largest denominator M, try to find distinct b, c < M
    # such that 1/b + 1/c = 5/31 - 1/M
    # Using divisor enumeration: if r = N/D, then (Nb-D)(Nc-D) = D^2
    # We need b, c distinct, positive, <= M
    
    best_max_denom = float('inf')
    best_solution = None
    
    # M must be at least ceil(31/5) = 7 for 1/M < 5/31
    # Also, we can bound M: since we're minimizing max denominator,
    # start from small M and go up until we find a valid solution
    for M in range(7, 300):
        r = target - Fraction(1, M)
        if r <= 0:
            continue
            
        N = r.numerator
        D = r.denominator
        
        # We need 1/b + 1/c = N/D where b, c are distinct positive integers <= M
        # (Nb - D)(Nc - D) = D^2
        # Let X = Nb - D, Y = Nc - D, so X * Y = D^2
        # b = (X + D) / N, c = (Y + D) / N
        # Both must be integers, distinct, positive, and <= M
        # Also b != c (distinct denominators)
        
        D_sq = D * D
        
        # Enumerate divisors of D_sq
        for i in range(1, int(math.isqrt(D_sq)) + 1):
            if D_sq % i != 0:
                continue
            X = i
            Y = D_sq // i
            
            # b = (X + D) / N must be integer
            if (X + D) % N != 0:
                continue
            b = (X + D) // N
            
            # c = (Y + D) / N must be integer
            if (Y + D) % N != 0:
                continue
            c = (Y + D) // N
            
            # Check constraints: b, c must be positive, distinct, and <= M
            if b <= 0 or c <= 0:
                continue
            if b == c:
                continue
            if b > M or c > M:
                continue
            
            # Also need b != M and c != M (since M is the largest, and b, c should be < M
            # or at most equal to M, but if one equals M that's fine since M is the max)
            # Actually, b and c can be equal to M? No, because then we'd have 1/M + 1/b = r
            # and we already subtracted 1/M. So b and c should be different from M.
            # Wait, the problem says "distinct denominators", so all three must be distinct.
            # M is one denominator, so b != M and c != M.
            if b == M or c == M:
                continue
            
            # Valid solution found with max denominator M
            best_max_denom = M
            best_solution = [M, b, c]
            break
        
        if best_solution:
            break
    
    if best_solution:
        return best_solution
    else:
        # Fallback: use greedy Egyptian fraction expansion
        r, values = Fraction(5, 31), []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
