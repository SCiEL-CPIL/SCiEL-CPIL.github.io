def construct():
    from fractions import Fraction
    import time
    import math
    
    deadline = time.time() + 85  # 85s deadline with 5s margin
    
    target = Fraction(5, 31)
    
    # Structural change: Use number-theoretic factorization approach.
    # For 1/a + 1/b + 1/c = 5/31, multiply by 31abc:
    # 31bc + 31ac + 31ab = 5abc
    # Rearrange: 5abc - 31ab - 31ac - 31bc = 0
    # Add (31^3)/5 to both sides? Let's use substitution.
    # Let x = 5a - 31, y = 5b - 31, z = 5c - 31
    # Then a = (x+31)/5, b = (y+31)/5, c = (z+31)/5
    # Substitute into 31bc + 31ac + 31ab = 5abc:
    # 31*(y+31)(z+31)/25 + 31*(x+31)(z+31)/25 + 31*(x+31)(y+31)/25 = 5*(x+31)(y+31)(z+31)/125
    # Multiply by 125:
    # 155*(y+31)(z+31) + 155*(x+31)(z+31) + 155*(x+31)(y+31) = 5*(x+31)(y+31)(z+31)
    # 31*5*[(y+31)(z+31) + (x+31)(z+31) + (x+31)(y+31)] = 5*(x+31)(y+31)(z+31)
    # 31*[(y+31)(z+31) + (x+31)(z+31) + (x+31)(y+31)] = (x+31)(y+31)(z+31)
    # Let X=x+31, Y=y+31, Z=z+31. Then:
    # 31*(YZ + XZ + XY) = XYZ
    # XYZ - 31XY - 31XZ - 31YZ = 0
    # Add 31^2: (XY - 31X - 31Y + 31^2)(Z) ... this is getting complex.
    # 
    # Alternative: just do efficient enumeration with the key insight that
    # for fixed a, the equation in b,c is:
    # 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # Let p = 5a-31, q = 31a. Need 1/b + 1/c = p/q.
    # (b+c)/(bc) = p/q => q(b+c) = pbc => pbc - qb - qc = 0
    # Multiply by p: p^2 bc - pqb - pqc = 0
    # (pb - q)(pc - q) = q^2
    # So let d1 = pb - q, d2 = pc - q, with d1*d2 = q^2
    # Then b = (d1+q)/p, c = (d2+q)/p
    # We need d1, d2 to be positive divisors of q^2, d1 <= d2,
    # and (d1+q) % p == 0, (d2+q) % p == 0, and b,c integers > a.
    
    best_solution = None
    best_max = 10000001
    
    # Enumerate a from 7 upward (since 1/a < 5/31 requires a > 6.2)
    # For each a, compute p = 5a - 31, q = 31a
    # Find all factor pairs (d1, d2) of q^2 with d1*d2 = q^2, d1 <= d2
    # Check if b = (d1+q)/p and c = (d2+q)/p are valid integers
    
    for a in range(7, 200):
        if time.time() > deadline:
            break
        
        p = 5 * a - 31
        q = 31 * a
        
        if p <= 0:
            continue
        
        # q^2 can be large. Find all divisors of q^2.
        # q = 31*a, so q^2 = 31^2 * a^2
        # Number of divisors of q^2 depends on factorization of a.
        # For efficiency, just enumerate d1 up to q (since d1 <= d2 and d1*d2=q^2, d1 <= q)
        # But q can be up to 31*200 = 6200, so q^2 ~ 38M. Enumerating divisors is fine.
        
        q_squared = q * q
        
        # Find all divisors of q_squared
        # Since we need d1 <= d2, d1 <= q
        # Iterate d1 from 1 to q, check if d1 divides q_squared
        # This is O(q) per a, and a goes up to 200, so total ~ sum of a from 7 to 200 * 31 ~ manageable
        
        found_for_a = False
        for d1 in range(1, q + 1):
            if time.time() > deadline:
                break
            if q_squared % d1 != 0:
                continue
            d2 = q_squared // d1
            if d2 < d1:
                continue
            
            # b = (d1 + q) / p, c = (d2 + q) / p
            num_b = d1 + q
            num_c = d2 + q
            
            if num_b % p != 0:
                continue
            if num_c % p != 0:
                continue
            
            b = num_b // p
            c = num_c // p
            
            # Check validity: b, c are positive integers, distinct from a and each other,
            # and 1/b + 1/c = p/q (should be guaranteed by construction)
            if b <= 0 or c <= 0:
                continue
            if b == a or c == a:
                continue
            if b == c:
                continue
            # Ensure a < b < c for canonical form (or just check distinctness)
            # Actually we just need distinct and positive
            if b < a or c < a:
                # Still valid as long as distinct, but let's ensure ordering for clarity
                pass
            
            # Verify: 1/a + 1/b + 1/c == 5/31
            s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
            if s != target:
                continue
            
            max_val = max(a, b, c)
            if max_val < best_max:
                best_max = max_val
                best_solution = [a, b, c]
                found_for_a = True
        
        if found_for_a and best_max <= a:
            # Can't do better with larger a since max >= a
            break
    
    if best_solution:
        return best_solution
    
    # Fallback: greedy
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
