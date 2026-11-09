"""Rewritten program with a stronger divisor-enumeration strategy."""
# EVOLVE-BLOCK-START
import math
from functools import reduce

def _factorize(num):
    """Return prime factorization of num as a dict {prime: exponent}."""
    if num <= 1:
        return {}
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

def _divisors_from_factorization(factors):
    """Generate all divisors from a prime factorization dict."""
    primes = list(factors.keys())
    exponents = list(factors.values())
    divs = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * power)
            power *= p
        divs = new_divs
    return divs

def _get_divisors(num):
    """Get all divisors of num."""
    if num == 0:
        return [0]
    factors = _factorize(num)
    return _divisors_from_factorization(factors)

def construct():
    answers = []
    BOUND = 1000000000000  # 10^12
    
    for n in range(3, 257):
        found = None
        # Strategy: Iterate x, use divisor enumeration on (B')^2
        x_start = n // 4 + 1
        # Limit x search to keep within time, but divisor method is fast so we can go further
        # If g is large, B' is small.
        # Let's try x up to n + 100 or so.
        for x in range(x_start, x_start + 500):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            g = math.gcd(a, b)
            a_prime = a // g
            b_prime = b // g
            
            # We need (a_prime * y - b_prime) * (a_prime * z - b_prime) = (b_prime)^2
            # Let u = a_prime * y - b_prime. Then u | (b_prime)^2.
            # u == -b_prime (mod a_prime)
            # y = (u + b_prime) / a_prime
            # z = ((b_prime)^2 // u + b_prime) / a_prime
            
            b_sq = b_prime * b_prime
            # If b_prime is too large, factorization might be slow?
            # b_prime <= n*x/g. Max n=256, x~500, b~128000.
            # Factorization of 128000 is fast.
            divs = _get_divisors(b_sq)
            
            # To speed up, we can filter divisors by congruence
            # u % a_prime == (-b_prime) % a_prime
            target_mod = (-b_prime) % a_prime
            
            for u in divs:
                if u % a_prime != target_mod:
                    continue
                
                # Check y
                num_y = u + b_prime
                if num_y % a_prime != 0:
                    continue
                y = num_y // a_prime
                
                # Check z
                v = b_sq // u
                num_z = v + b_prime
                if num_z % a_prime != 0:
                    continue
                z = num_z // a_prime
                
                # Check constraints
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                # Verify equation
                # 1/x + 1/y + 1/z = 4/n
                # n(yz + xz + xy) = 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
