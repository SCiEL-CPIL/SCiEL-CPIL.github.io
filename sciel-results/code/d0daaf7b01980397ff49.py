import random
import math
import time

# Miller-Rabin Primality Test
def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    # Write n-1 as d*2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Witnesses for 32-bit and 64-bit integers
    # For n < 3,474,749,660,383, these bases are sufficient
    # Our n <= 10^12, so this is more than enough.
    witnesses = [2, 3, 5, 7, 11, 13]
    
    for a in witnesses:
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

# Pollard's Rho Algorithm
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

# Sieve of Eratosthenes for small primes
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

SMALL_PRIMES = sieve(10000)

def factor(n, result=None):
    if result is None:
        result = []
    if n == 1:
        return result
    if is_prime(n):
        result.append(n)
        return result
    
    # Try small primes first
    for p in SMALL_PRIMES:
        if n % p == 0:
            result.append(p)
            return factor(n // p, result)
            
    d = pollard_rho(n)
    factor(d, result)
    factor(n // d, result)
    return result

def phi(n):
    if n == 1:
        return 1
    factors = factor(n)
    # Get unique prime factors
    unique_primes = set(factors)
    result = n
    for p in unique_primes:
        result -= result // p
    return result

def construct():
    random.seed(42) # Deterministic
    max_start = 10**12 - 2000
    min_start = 10**11
    
    best_start = 1
    best_len = 1
    
    # Scan multiple random intervals
    # Each interval length 2000 allows finding max k up to 200
    num_intervals = 50 
    interval_len = 2000
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    for _ in range(num_intervals):
        if time.time() - start_time > time_limit:
            break
            
        s = random.randint(min_start, max_start)
        
        # Compute phi for the interval
        phis = []
        for i in range(interval_len):
            phis.append(phi(s + i))
            
        # Find longest subarray with distinct values
        seen = {}
        left = 0
        current_max = 0
        current_max_start = 0
        
        for right in range(interval_len):
            val = phis[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            
            if right - left + 1 > current_max:
                current_max = right - left + 1
                current_max_start = left
                
        if current_max > best_len:
            best_len = current_max
            best_start = s + current_max_start
            
        # Early exit if we found the maximum possible k
        if best_len == 200:
            break

    return {'start': best_start, 'length': best_len}
