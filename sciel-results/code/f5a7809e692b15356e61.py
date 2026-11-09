"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # The known optimal solution for 5/31 minimizing the largest denominator.
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    # Largest denominator is 248.
    # This is the global minimum for the largest denominator in any Egyptian fraction expansion of 5/31.
    
    candidates = [8, 31, 248]
    
    # Verification (optional but safe)
    s = sum(Fraction(1, d) for d in candidates)
    if s != Fraction(5, 31):
        # Fallback to seed behavior if verification fails (should not happen)
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return candidates
# EVOLVE-BLOCK-END
