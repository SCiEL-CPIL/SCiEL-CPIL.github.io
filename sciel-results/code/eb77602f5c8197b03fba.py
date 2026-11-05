import math

def find_solution(n):
    x_min = n // 4 + 1
    x_max = 3 * n // 4
    
    for x in range(x_min, x_max + 1):
        A = 4 * x - n
        B = n * x
        
        g = math.gcd(A, B)
        A_prime = A // g
        B_prime = B // g
        
        # We need d | B_prime such that:
        # 1. (d + B_prime) % A_prime == 0
        # 2. y = (d + B_prime) // A_prime > x
        # 3. z = B_prime * y // d is integer and z > y
        #    z is integer iff B_prime * y % d == 0.
        #    Since y = (d+B_prime)/A_prime,
        #    z = B_prime (d+B_prime) / (A_prime d)
        #    We know (d+B_prime) is div by A_prime.
        #    Let y = k. z = B_prime k / d.
        #    We need d | B_prime k.
        #    Since gcd(d, B_prime) might not be 1, this is not trivial.
        #    But we can just check if (B_prime * y) % d == 0.
        
        # Also y > x implies d + B_prime > x * A_prime
        # d > x * A_prime - B_prime
        # Let limit = x * A_prime - B_prime
        # We need d > limit.
        
        # And z > y implies B_prime / d > 1 => d < B_prime.
        
        limit = x * A_prime - B_prime
        
        # Iterate divisors of B_prime
        # To be efficient, we can just iterate d from max(1, limit+1) to B_prime?
        # No, that's the old way.
        # We should iterate divisors.
        
        # However, checking all d up to sqrt(B_prime) is fast.
        # We need to find d such that d | B_prime and d > limit and d < B_prime.
        # And (d + B_prime) % A_prime == 0.
        
        # Let's collect valid d's or just return first solution.
        
        # Optimization: 
        # d must be a divisor of B_prime.
        # Let's iterate i from 1 to isqrt(B_prime).
        
        # But we need d > limit.
        # If limit is large, we might miss small divisors.
        # We can just check all divisors.
        
        isqrt_bp = math.isqrt(B_prime)
        
        for d in range(1, isqrt_bp + 1):
            if B_prime % d != 0:
                continue
            
            # Check d1 = d
            # Check d2 = B_prime // d
            
            # We need to check both d and B_prime//d as potential divisors 'd' in the formula.
            # The variable 'd' in my derivation is the divisor corresponding to the term (Ay-B).
            # Let's call the divisor 'div'.
            
            # Candidate div1 = d
            if d > limit and d < B_prime:
                if (d + B_prime) % A_prime == 0:
                    y = (d + B_prime) // A_prime
                    if y > x:
                        num = B_prime * y
                        if num % d == 0:
                            z = num // d
                            if z > y:
                                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                                    return [x, y, z]
            
            # Candidate div2 = B_prime // d
            d2 = B_prime // d
            if d2 != d:
                if d2 > limit and d2 < B_prime:
                    if (d2 + B_prime) % A_prime == 0:
                        y = (d2 + B_prime) // A_prime
                        if y > x:
                            num = B_prime * y
                            if num % d2 == 0:
                                z = num // d2
                                if z > y:
                                    if x <= 10**12 and y <= 10**12 and z <= 10**12:
                                        return [x, y, z]
                                        
    return None

_TABLE = {}
for _n in range(3, 257):
    _TABLE[_n] = find_solution(_n)

def construct():
    return [_TABLE[n] for n in range(3, 257)]
