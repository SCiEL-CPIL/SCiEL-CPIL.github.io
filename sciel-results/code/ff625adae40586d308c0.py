import math
import time
import random

# Precompute small primes for trial division
def _gen_primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

_SMALL_PRIMES = _gen_primes_upto(10000)

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return n in (2, 3)
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    """Brent's variant of Pollard's rho."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    y, c, m = 2, 1, 128
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    return g

def largest_prime_factor(n, limit=None):
    """Returns the largest prime factor of n. If limit is provided, returns > limit if any factor > limit exists, else limit."""
    if n == 1:
        return 1
    if n < 0:
        n = -n
    
    # Trial division by small primes
    max_f = 1
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        if limit and p > limit:
            break
        if n % p == 0:
            if p > max_f:
                max_f = p
            while n % p == 0:
                n //= p
        if limit and max_f > limit:
            return max_f

    if n == 1:
        return max_f
    if is_prime(n):
        if n > max_f:
            max_f = n
        if limit and max_f > limit:
            return max_f
        return max_f

    # Recursively find factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
            if limit and max_f > limit:
                return max_f
        else:
            d = pollard_brent(m)
            if d == m:
                if m > max_f:
                    max_f = m
                if limit and max_f > limit:
                    return max_f
            else:
                stack.append(d)
                stack.append(m // d)
    return max_f

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    best_k = 0
    best_p = 2

    # Sieve primes up to 10^7
    limit = 10_000_000
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]

    # Iterate through primes
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick rejection: if p^2+1 is prime, k=0
        # p^2+1 is even. (p^2+1)/2 is integer.
        # If (p^2+1)/2 is prime, then p^2+1 has a factor > p (since (p^2+1)/2 > p for p>2).
        # So k=0.
        half_n1 = (p2 + 1) // 2
        if is_prime(half_n1):
            continue
            
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p2 + i
            
            # Check if n_val is prime
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k += 1
                    continue
            
            # Find largest prime factor
            # Optimization: if we just need to know if max factor > p
            lf = largest_prime_factor(n_val, limit=p)
            
            if lf > p:
                break
            
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good k, we might want to stop early if time is low
            if best_k >= 15 and time.time() > deadline - 30:
                break

    # Try some larger primes beyond the sieve with targeted search
    if time.time() < deadline - 20:
        random.seed(42)
        # Try primes in ranges where p^2+1 might have small factors
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Generate candidate primes in various ranges
            range_low = random.choice([10_000_000, 50_000_000, 100_000_000, 500_000_000])
            range_high = min(range_low + 10_000_000, 10**9)
            p_candidate = random.randint(range_low, range_high)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if not is_prime(p_candidate):
                continue
            p = p_candidate
            p2 = p * p
            n1 = p2 + 1
            if is_prime(n1):
                continue
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                n_val = p2 + i
                if is_prime(n_val):
                    if n_val > p:
                        break
                    else:
                        k += 1
                        continue
                lf = largest_prime_factor(n_val, limit=p)
                if lf > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
