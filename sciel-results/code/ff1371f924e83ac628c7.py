import math
import time
import random
import sys

# Deterministic Miller-Rabin bases for n < 2^64
# For our range (n up to 10^18), the first 12 primes are sufficient.
# Actually, for n < 3,474,749,660,383, bases 2,3,5,7,11,13,17 are enough.
# Our n can be up to 10^18.
# A robust set for 64-bit is {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}.
# Or we can use the specific set for 2^64: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67}
# Let's use the first 12 primes for speed and reliability.

_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

# Pre-computed small primes for trial division
_SMALL_PRIMES = []
def _init_small_primes(limit=1000):
    if _SMALL_PRIMES:
        return
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    _SMALL_PRIMES.extend([i for i, prime in enumerate(is_prime_arr) if prime])
_init_small_primes(1000)

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    # Random parameters
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g = 1
        r = 1
        q = 1
        x = y
        ys = y
        
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        
        if g != n:
            return g

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    Returns False otherwise.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes
    for sp in _SMALL_PRIMES:
        if sp > p:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
            if n == 1:
                return True
    
    if n == 1:
        return True
    
    # If remaining n is prime, it must be <= p
    if is_prime(n):
        return n <= p
    
    # Otherwise, we need to factor n further.
    # We can use Pollard-Brent to find a factor d.
    # If d > p, then n is not p-smooth.
    # If d <= p, we can divide n by d and continue.
    # But Pollard-Brent might return a composite factor.
    # So we need to recursively check.
    
    # To avoid deep recursion and repeated work, let's just find the largest prime factor.
    # Or, we can try to find a factor > p.
    
    # Strategy:
    # 1. Find a factor d of n using Pollard-Brent.
    # 2. If d > p, return False.
    # 3. If d <= p, divide n by d and check if the remaining n is p-smooth.
    #    But d might be composite.
    
    # Better strategy:
    # Find the largest prime factor of n.
    # If it is <= p, return True.
    
    # Let's implement a function to find the largest prime factor.
    
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        if is_prime(x):
            return x
        
        # Trial division
        for sp in _SMALL_PRIMES:
            if sp > x:
                break
            if x % sp == 0:
                # Divide out all sp
                while x % sp == 0:
                    x //= sp
                if x == 1:
                    return sp
                # Check remaining x
                return max(sp, largest_prime_factor(x))
        
        # Pollard-Brent
        d = pollard_brent(x)
        return max(largest_prime_factor(d), largest_prime_factor(x // d))
    
    max_pf = largest_prime_factor(n)
    return max_pf <= p

def sieve(limit):
    if limit < 2:
        return []
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_p = 7
    best_k = 1
    
    # Strategy:
    # 1. Sieve primes up to 10^6.
    # 2. For each prime p, check the run length k.
    # 3. Use is_p_smooth for efficiency.
    # 4. Early exit if k <= best_k.
    
    sieve_limit = 10**6
    primes = sieve(sieve_limit)
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check run length
        k = 0
        p2 = p * p
        
        # We want to find the largest k such that p^2+1 ... p^2+k are all p-smooth.
        # Optimization:
        # If we have already found best_k, we can stop checking for this p
        # as soon as we find a non-smooth number at position j <= best_k.
        
        valid_so_far = True
        for j in range(1, best_k + 1):
            n = p2 + j
            if not is_p_smooth(n, p):
                valid_so_far = False
                break
        
        if not valid_so_far:
            continue
        
        # All 1..best_k are smooth.
        # Now extend k
        current_k = best_k
        while current_k < 200:
            current_k += 1
            n = p2 + current_k
            
            if not is_p_smooth(n, p):
                break
            
            if current_k > best_k:
                best_k = current_k
                best_p = p
                
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
