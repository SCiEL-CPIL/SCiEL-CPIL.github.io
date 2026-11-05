import time
import math

# Precompute primes up to a limit sufficient for factorizing B = n*x
# n <= 256, x <= 3n/4 <= 192. Max B = 256 * 192 = 49152.
MAX_B = 50000

def sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    return primes

PRIMES = sieve_primes(MAX_B)

def factorize(num):
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

def get_divisors_from_factorization(prime_exp_pairs):
    """
    Generate all divisors from a list of (prime, exponent) pairs.
    """
    divisors = [1]
    for p, e in prime_exp_pairs:
        current_len = len(divisors)
        pow_val = 1
        for _ in range(e):
            pow_val *= p
            for i in range(current_len):
                divisors.append(divisors[i] * pow_val)
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254
    
    # Cache for factorizations to avoid re-computing for same B
    factor_cache = {}
    divisor_cache = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
            
        solved = False
        # x must satisfy 4x - n > 0, so x >= n//4 + 1
        # Also x should be < 3n/4 for distinct solutions with x<y<z
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            # A must be positive
            if A <= 0:
                continue
                
            # Get factors of B
            if B not in factor_cache:
                factor_cache[B] = factorize(B)
            factors_B = factor_cache[B]
            
            # We need divisors of B^2.
            # If B = prod p^e, B^2 = prod p^(2e)
            factors_B2 = [(p, 2*e) for p, e in factors_B]
            
            # Generate divisors of B^2
            # Optimization: We only care about divisors u where u < B
            # to ensure y > x (since y = (u + B)/A and we want y > x)
            if B not in divisor_cache:
                divisor_cache[B] = get_divisors_from_factorization(factors_B2)
            divs = divisor_cache[B]
            
            for u in divs:
                # u must be < B for y > x
                if u >= B:
                    continue
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                # Check if y is integer (guaranteed by mod check)
                # Check y > x and y > 0
                if y <= x or y <= 0:
                    continue
                
                v = (B * B) // u
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                
                # Check distinctness: x, y, z all distinct
                # and z > y for ordering
                if z <= y or z == x or y == x:
                    continue
                
                # Check bound z <= 10^12
                if z > 10**12:
                    continue
                
                # All checks passed
                answers[n - 3] = [x, y, z]
                solved = True
                break
            if solved:
                break
                
    return answers
