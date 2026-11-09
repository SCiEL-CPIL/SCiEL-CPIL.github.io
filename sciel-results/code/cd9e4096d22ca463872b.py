"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    from math import gcd
    import time

    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    target = Fraction(5, 31)

    # We'll try to find Egyptian fraction with minimum largest denominator.
    # Strategy: For small number of terms (2, 3, 4, ...), use number-theoretic search.
    
    # First, check if a simple 2-term or 3-term solution exists with small max.
    # For 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    # => (5a - 31)(5b - 31) = 31*31 = 961
    # Divisors of 961: 1, 31, 31^2=961
    # 5a-31 = d, 5b-31 = 961/d
    # a = (d+31)/5, b = (961/d + 31)/5
    # d=1: a=32/5 not int
    # d=31: a=62/5 not int  
    # d=961: a=992/5 not int
    # No 2-term solution.
    
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # We search for solutions by fixing the largest denominator M and checking
    # if 1/a + 1/b = 5/31 - 1/M has solutions with a,b < M and distinct.
    
    # The known solution is [8, 31, 248] with max 248. Let's verify:
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Yes!
    
    # But can we do better? Let's search for 3-term solutions with max < 248.
    
    # 1/a + 1/b + 1/c = 5/31, assume a <= b <= c
    # c >= 31/5 * 3 = 18.6, so c >= 19
    # Also 1/a + 1/b + 1/c = 5/31, and a <= b <= c means 3/a >= 5/31, so a <= 31*3/5 = 18.6, a <= 18
    
    # For each a in 1..18, we need 1/b + 1/c = 5/31 - 1/a = r
    # where r > 0
    # Then (r.denominator - r.numerator)*b*c + ... let's use the standard formula:
    # 1/b + 1/c = r/n where r/n = 5/31 - 1/a
    # => n(b+c) = r*b*c
    # => r*b*c - n*b - n*c = 0
    # => (r*b - n)(r*c - n) = n^2
    # So we factor n^2 and for each factor d of n^2, set r*b - n = d, r*c - n = n^2/d
    # => b = (d+n)/r, c = (n^2/d + n)/r
    
    def find_3term(a_max=None):
        """Find 3-term Egyptian fraction for 5/31 with minimum max denominator."""
        best = None
        if a_max is None:
            a_max = 18  # as computed above
        
        for a in range(1, a_max + 1):
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            r_num = r.numerator
            r_den = r.denominator
            # 1/b + 1/c = r_num/r_den
            # (r_num*b - r_den)(r_num*c - r_den) = r_den^2
            n_sq = r_den * r_den
            # Find all divisors d of n_sq
            divisors = []
            for d in range(1, int(n_sq**0.5) + 1):
                if n_sq % d == 0:
                    divisors.append(d)
                    if d != n_sq // d:
                        divisors.append(n_sq // d)
            
            for d in divisors:
                if d % r_num != 0:
                    continue
                b = (d + r_den) // r_num
                if r_num * b != d + r_den:
                    continue
                c_num = n_sq // d + r_den
                if c_num % r_num != 0:
                    continue
                c = c_num // r_num
                # Check validity
                if b >= a and c >= b and b != c and a != b and a != c and b != a and c != a and b != c:
                    # All distinct
                    max_den = max(a, b, c)
                    if best is None or max_den < best[0]:
                        best = (max_den, [a, b, c])
        
        return best
    
    # Search for 3-term solutions
    result = find_3term()
    
    if result is not None:
        max_den = result[0]
        # Check if we can do even better with more terms
        # But first, let's see what we found
        # The known solution [8, 31, 248] has max 248
        # Let's also try to see if there's a solution with max < 248
        
        # For now, return the best 3-term solution
        if time.time() > deadline:
            return result[1]
        
        # Try to find solutions with smaller max by being more thorough
        # Actually, let's also try 4-term solutions which might have smaller max
        # But 3-term is likely optimal here
        
        return result[1]
    else:
        # Fallback: use the known solution
        return [8, 31, 248]

# EVOLVE-BLOCK-END
