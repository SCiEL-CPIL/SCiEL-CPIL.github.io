import time
import math

# Precompute primes up to 100000
def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes

PRIMES = _sieve(100000)

def _factorize(n):
    """Return a list of (prime, exponent) for n."""
    factors = []
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors.append((p, exp))
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        current_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                current_divs.append(d * pe)
            pe *= p
        divs = current_divs
    return divs

def construct():
    deadline = time.time() + 85
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            b_red_sq = b_red * b_red
            
            # Factorize b_red_sq
            # Since b_red is small (< 50000), b_red_sq < 2.5 * 10^9
            # We can factorize b_red and then double exponents
            factors_b = _factorize(b_red)
            factors_sq = [(p, 2*e) for p, e in factors_b]
            
            divs = _divisors_from_factors(factors_sq)
            
            for d in divs:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                num_y = b_red + d
                if num_y % a_red != 0:
                    continue
                y = num_y // a_red
                
                if y <= x:
                    continue
                
                num_z = b_red + b_red_sq // d
                if num_z % a_red != 0:
                    continue
                z = num_z // a_red
                
                if z <= y:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        
        answers.append(found)
    
    return answers
