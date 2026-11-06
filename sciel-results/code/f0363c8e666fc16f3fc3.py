"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # Search for Egyptian fraction of 5/31 minimizing the largest denominator
    # Try 3 terms: 1/a + 1/b + 1/c = 5/31, a < b < c
    
    best_solution = None
    best_max = 10000001
    
    # a >= 7 since 1/6 < 5/31
    for a in range(7, 500):
        # For given a, 1/b + 1/c = 5/31 - 1/a
        # Let r = 5/31 - 1/a = (5a-31)/(31a)
        # Need r > 0, so a > 31/5 = 6.2, i.e., a >= 7
        # 1/b + 1/c = (5a-31)/(31a)
        # (b+c)/(bc) = (5a-31)/(31a)
        # 31a(b+c) = (5a-31)bc
        # (5a-31)b - 31a = (5a-31)bc - 31ab... let me use the divisor method
        # (p*b - q)(p*c - q) = q^2 where p = 5a-31, q = 31a
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        
        q2 = q * q
        # Find all divisors of q2
        # For each divisor d of q2, set 5a*b - q = d, so b = (d + q) / (5a)
        # and 5a*c - q = q2/d, so c = (q2/d + q) / (5a)
        # Need d | q2, and (d+q) % (5a) == 0, and (q2/d + q) % (5a) == 0
        # Also need b > a and c > b
        
        # Iterate over divisors d of q2
        # q2 can be large, so let's be smart
        # Actually for small a, q2 is manageable
        
        # Let's just iterate b from a+1 upward and solve for c
        # 1/c = (5a-31)/(31a) - 1/b = ((5a-31)*b - 31a) / (31a*b)
        # c = 31a*b / ((5a-31)*b - 31a)
        
        for b in range(a + 1, 5000):
            denom_val = p * b - q
            if denom_val <= 0:
                continue
            
            numerator_val = q * b
            
            if numerator_val % denom_val != 0:
                continue
            
            c = numerator_val // denom_val
            
            if c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: known solution
    return [7, 62, 434]
# EVOLVE-BLOCK-END
