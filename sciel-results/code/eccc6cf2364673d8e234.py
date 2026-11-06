import math
import time

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
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

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization dict."""
    divisors = [1]
    for p, e in factors.items():
        current = []
        pe = 1
        for _ in range(e + 1):
            for d in divisors:
                current.append(d * pe)
            pe *= p
        divisors = current
    return divisors

def construct():
    deadline = time.time() + 85  # 85s safety margin (evaluator kills at 90s)
    answers = [None] * 254  # n=3..256
    
    # Precompute factorizations for small numbers to speed up
    # B = n * x, n in 3..256, x in ~n/4..3n/4, so B up to ~256*192 = 49152
    # We'll factorize B on the fly but cache results
    
    factor_cache = {}
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: 1/x < 4/n => x > n/4
        # and 3/x >= 4/n => x <= 3n/4 (for x <= y <= z to be possible)
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (A*y - B)(A*z - B) = B^2
            # So A*y - B must be a divisor d of B^2
            # y = (d + B) / A must be positive integer
            # z = (B^2/d + B) / A must be positive integer
            # Also need y != x, z != x, y != z, and all <= 10^12
            
            # Factorize B to get divisors of B^2
            if B not in factor_cache:
                factor_cache[B] = _factorize(B)
            b_factors = factor_cache[B]
            
            # B^2 has factors with doubled exponents
            b2_factors = {p: 2*e for p, e in b_factors.items()}
            b2_divisors = _divisors_from_factors(b2_factors)
            
            # For each divisor d of B^2, check validity
            # We want y < z, so we can restrict d <= B (since A*y-B = d and A*z-B = B^2/d,
            # if d <= B then y <= z)
            # Actually let's just check all and pick valid ones
            
            for d in b2_divisors:
                # y = (d + B) / A
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= 0 or y <= x:
                    continue
                
                # z = (B^2//d + B) / A
                num_z = B * B // d + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= 0 or z <= y:
                    continue
                
                # Check distinctness: x, y, z all distinct
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if x > 1000000000000 or y > 1000000000000 or z > 1000000000000:
                    continue
                
                # Verify the equation
                # 1/x + 1/y + 1/z = 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
