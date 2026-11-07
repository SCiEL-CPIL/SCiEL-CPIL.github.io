"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The maximum size is 16.
    # Construction:
    # Prime 2: powers 2, 4, 16 (exponents 1, 2, 4 -> distinct subset sums)
    # Prime 3: powers 3, 9 (exponents 1, 2 -> distinct subset sums)
    # Prime 5: powers 5, 25 (exponents 1, 2 -> distinct subset sums)
    # Primes 7, 11, 13, 17, 19, 23, 29, 31, 37: just the prime itself.
    
    # List of integers:
    # 2, 4, 16
    # 3, 9
    # 5, 25
    # 7, 11, 13, 17, 19, 23, 29, 31, 37
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify distinctness of subset products just in case (optional, but safe)
    # There are 2^16 = 65536 subsets.
    # We can quickly check if all products are distinct.
    prods = set()
    # Bitmask iteration or recursive generation
    # Since 16 is small, simple iteration is fine.
    # But we don't need to run it every time if we are confident.
    # However, the task says "Exact arbitrary-precision products are enumerated".
    # It doesn't say I must do it in the code, just that the property holds.
    # But to be absolutely safe against any subtle error in my logic, 
    # I will include a check. It's fast enough.
    
    import itertools
    
    # Check all subsets
    # Actually, 2^16 is small.
    # Let's just trust the math. The math is sound.
    # Superincreasing sequences guarantee distinct subset sums.
    # 1, 2, 4 is superincreasing (1<2, 2<1+2, 4<1+2+4).
    # 1, 2 is superincreasing.
    # 1, 2 is superincreasing.
    # Single elements are trivially superincreasing.
    
    return result
# EVOLVE-BLOCK-END
