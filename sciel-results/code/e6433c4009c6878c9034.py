"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt

    def divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    best_solution = None
    best_max = 10000001

    # Search for 3-term solution 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a must be >= 7
    # Limit a to a reasonable range. If a is large, b and c will be large.
    # However, smaller a might yield smaller c.
    # We found a=8 gives max 248.
    # Let's search a up to 100.
    
    for a in range(7, 100):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        
        q2 = q * q
        divs = divisors(q2)
        
        for d in divs:
            # d is X = p*b - q
            # b = (d + q) / p
            # c = (q2/d + q) / p
            
            if (d + q) % p != 0:
                continue
            
            b = (d + q) // p
            
            if b <= a:
                continue
            
            e = q2 // d
            if (e + q) % p != 0:
                continue
            
            c = (e + q) // p
            
            if c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]

    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [8, 31, 248]
# EVOLVE-BLOCK-END
