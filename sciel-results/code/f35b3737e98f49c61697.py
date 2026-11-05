import time
from math import gcd

def construct():
    answers = []
    deadline = time.time() + 50  # 50s safety margin
    
    # Precompute primes up to 1000 for fast factorization?
    # Or just trial division since B is small.
    
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors

    def get_divisors_from_factors(factors):
        # factors is a dict {prime: exponent}
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        found = None
        # x must satisfy n/4 < x < 3n/4
        # 1/x + 1/y + 1/z = 4/n
        # If x <= n/4, 1/x >= 4/n, so 1/y + 1/z <= 0, impossible for positive y,z.
        # If x >= 3n/4, 1/x <= 4/(3n). 1/y + 1/z >= 4/n - 4/(3n) = 8/(3n).
        # 1/y + 1/z < 2/x <= 8/(3n) => x < 3n/4.
        # So x is in (n/4, 3n/4).
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B
            # u * v = B^2
            # y = (u + B) / A
            # z = (v + B) / A
            
            # Factorize B to get divisors of B^2
            # B = n * x
            # Factorize n and x, combine exponents, double them for B^2
            
            # Simpler: factorize B directly
            b_factors = factorize(B)
            b_sq_factors = {p: 2*e for p, e in b_factors.items()}
            
            divs = get_divisors_from_factors(b_sq_factors)
            
            # We can sort divisors to break early if y > z?
            # Or just iterate all.
            # To ensure y != z, we can just check u != v.
            # To ensure y, z > 0, u, v > -B. Since u, v are divisors of B^2, they are positive.
            
            for u in divs:
                # Check if y is integer
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                
                # Calculate v
                v = (B * B) // u
                
                # Check if z is integer
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                
                # Check distinctness and bounds
                if y == z:
                    continue
                if y == x or z == x:
                    continue
                if y <= 0 or z <= 0:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # We found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
