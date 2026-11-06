import math
import time

# Precompute small primes for trial division and Miller-Rabin
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_prime(n):
    if n < 2:
        return False
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
    # For n < 3,825,123,056,546,413,051, first 12 primes suffice
    for a in _SMALL_PRIMES:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                x = 3
                y = 3
                c = 1
    return d

def largest_prime_factor(n):
    if n == 1:
        return 1
    if n < 0:
        n = -n
    # Trial division by small primes
    factors = []
    for p in _SMALL_PRIMES:
        if n % p == 0:
            while n % p == 0:
                factors.append(p)
                n //= p
            if n == 1:
                return max(factors)
    # Recursively find factors
    stack = [n]
    max_f = 1
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    return max_f

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    best_k = 0
    best_p = 2

    # Sieve primes up to 10^7 to get a good list of candidate primes
    # 10^7 / ln(10^7) ≈ 664k primes, which is manageable
    limit = 10**7
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]

    # We'll iterate through primes in increasing order
    # For each prime p, check how many consecutive numbers p^2+1, p^2+2, ... are p-smooth
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Quick rejection: if p is very small, max k is limited
        # For p=2: p^2+1=5 (prime > 2), k=0
        # For p=3: p^2+1=10=2*5 (5>3), k=0
        # For p=5: p^2+1=26=2*13 (13>5), k=0
        # For p=7: p^2+1=50=2*5*5 (5<=7), p^2+2=51=3*17 (17>7), k=1
        # So small primes have small k. We can skip some.
        
        # Check sequence
        k = 0
        # p^2 + 1
        # Optimization: if p^2+1 is prime, k=0
        n1 = p * p + 1
        if is_prime(n1):
            continue
            
        # Check smoothness for i=1 to 200
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p * p + i
            
            # Quick check: is n_val prime? If so and n_val > p, fail
            if is_prime(n_val):
                if n_val > p:
                    break
                else:
                    k += 1
                    continue
            
            # Find largest prime factor
            lf = largest_prime_factor(n_val)
            
            if lf > p:
                break
            
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p

    # If we have time, try some larger primes beyond the sieve
    if time.time() < deadline - 20:
        # Try some random large primes
        import random
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            # Generate a random candidate in [10^7, 10^9]
            p_candidate = random.randint(10**7, 10**9)
            # Make it odd
            if p_candidate % 2 == 0:
                p_candidate += 1
            # Check if prime
            if is_prime(p_candidate):
                p = p_candidate
                k = 0
                n1 = p * p + 1
                if is_prime(n1):
                    continue
                for i in range(1, 201):
                    if time.time() > deadline:
                        break
                    n_val = p * p + i
                    if is_prime(n_val):
                        if n_val > p:
                            break
                        else:
                            k += 1
                            continue
                    lf = largest_prime_factor(n_val)
                    if lf > p:
                        break
                    k += 1
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
