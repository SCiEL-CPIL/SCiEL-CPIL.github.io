import time
import math

MAX_B = 256 * (256 + 512)  # safe upper bound for n(n+a) with a up to 512
MAX_B = max(MAX_B, 256 * 256 + 256 * 512)
MAX_B = 256 * 1024  # generous

def sieve_primes(limit):
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    return primes

PRIMES = sieve_primes(MAX_B)

def factorize(num):
    """Return list of (prime, exponent) for num."""
    if num <= 1:
        return []
    factors = []
    temp = num
    for p in PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    if temp > 1:
        factors.append((temp, 1))
    return factors

def divisors_of_square(factors):
    """
    Given factorization of B as list of (p, e), generate all divisors u of B^2.
    B^2 has factorization (p, 2e). u can have exponent 0..2e for each p.
    """
    # Build list of possible exponent choices per prime
    choices = []
    for p, e in factors:
        max_exp = 2 * e
        pows = [p**i for i in range(max_exp + 1)]
        choices.append(pows)
    
    # Generate all products via iterative expansion
    result = [1]
    for pows in choices:
        new_result = []
        for base in result:
            for pw in pows:
                new_result.append(base * pw)
        result = new_result
    return result

def solve_one(n, deadline):
    """Find distinct x,y,z for given n, or return None."""
    # x = n + a, a >= 1
    # A = 4a, B = n(n+a)
    # u*v = B^2, y = (u+B)/A, z = (v+B)/A
    # Need: A | (u+B), A | (v+B), y != z, y != x, z != x, y,z > 0, distinct
    
    for a in range(1, 2000):
        if time.time() > deadline:
            return None
        x = n + a
        A = 4 * a
        B = n * (n + a)
        # B = n*x
        # Need u | B^2, and (u + B) % A == 0, (B^2//u + B) % A == 0
        # Also y = (u+B)//A, z = (B^2//u + B)//A
        # y and z must be positive integers, distinct, and != x
        
        fac = factorize(B)
        divs = divisors_of_square(fac)
        
        # Sort divisors to try smaller u first (smaller y)
        # But we want efficiency; let's just iterate and check
        B2 = B * B
        
        for u in divs:
            if time.time() > deadline:
                return None
            if B2 % u != 0:
                continue
            v = B2 // u
            num_y = u + B
            num_z = v + B
            if num_y % A != 0:
                continue
            if num_z % A != 0:
                continue
            y = num_y // A
            z = num_z // A
            # Check positivity
            if y <= 0 or z <= 0:
                continue
            # Check distinctness: x, y, z all distinct
            if y == x or z == x or y == z:
                continue
            # Check bound
            if y > 10**12 or z > 10**12:
                continue
            # Verify: 1/x + 1/y + 1/z == 4/n
            # Cross multiply: yz*n + xz*n + xy*n == 4xyz
            lhs = y * z * n + x * z * n + x * y * n
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
    return None

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    answers = [None] * 254
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        result = solve_one(n, deadline)
        if result is not None:
            answers[idx] = result
    return answers
