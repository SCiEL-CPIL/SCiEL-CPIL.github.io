"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import multiprocessing as mp
import random
from functools import lru_cache

# Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
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

def factorize(n):
    if n == 1:
        return []
    factors = []
    # Trial division for small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
            continue
        d = pollard_rho(m)
        stack.append(d)
        stack.append(m // d)
    return factors

def largest_prime_factor(n):
    if n == 1:
        return 1
    # Quick check for small factors to avoid overhead if possible? 
    # Pollard Rho is fast enough.
    factors = factorize(n)
    return max(factors)

def check_prime(p, deadline):
    # p is a prime
    k = 0
    # We want to maximize k such that p^2 + 1 ... p^2 + k are p-smooth
    # Actually, the condition is "all prime factors at most p".
    # This is equivalent to saying p^2 + i is p-smooth.
    # Note: p^2 + 1 is always coprime to p.
    
    # Optimization: If p is large, p^2 is large.
    # We check sequentially.
    base = p * p
    while time.time() < deadline:
        k += 1
        n = base + k
        # If n > p, we need to check if its largest prime factor <= p
        # If n <= p, it's automatically p-smooth (since n is prime or 1, and n<=p)
        # But n starts at p^2+1, so n > p for p >= 2.
        
        # Optimization: If n has a prime factor > p, it fails.
        # We can check if n is divisible by any prime > p? No, that's hard.
        # We must factorize.
        
        # Early exit check: 
        # If n is prime and n > p, it fails.
        if is_prime(n):
            if n > p:
                break
            else:
                # This case n<=p is impossible for k>=1, p>=2
                pass
        else:
            max_pf = largest_prime_factor(n)
            if max_pf > p:
                break
        
        if k >= 200:
            break
            
    return p, k

def worker(args):
    p, deadline = args
    try:
        return check_prime(p, deadline)
    except Exception:
        return p, 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate candidate primes
    # We want to search a wide range.
    # Small primes might have long runs?
    # Let's test a mix of small and medium primes.
    # The seed program found k=2 for small primes.
    # Maybe larger primes have longer runs?
    
    # Strategy: 
    # 1. Check a few small primes quickly.
    # 2. Check random primes in increasing ranges.
    
    # Simple prime generator
    def primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]
    
    # Start with small primes
    small_primes = primes_up_to(1000)
    
    # Create tasks
    tasks = []
    for p in small_primes:
        tasks.append((p, deadline))
        
    # Add some larger primes
    # Random sampling in [1000, 100000]
    for _ in range(50):
        p = random.randint(1000, 100000)
        # Find next prime
        while not is_prime(p):
            p += 1
            if p > 100000:
                break
        tasks.append((p, deadline))
        
    # Use pool
    # Limit processes to avoid overhead on small tasks
    num_procs = min(4, mp.cpu_count())
    
    best_k = 0
    best_p = 2
    
    with mp.Pool(processes=num_procs) as pool:
        # Process in chunks to allow early stopping? 
        # Pool.map is blocking. 
        # We can use imap_unordered to get results as they finish.
        
        # To respect deadline, we should ideally not block on map.
        # But imap_unordered allows us to stop early if we want, 
        # though the pool will still run until closed.
        
        # Since deadline is 140s, and tasks are fast, 
        # we can just map and check time.
        
        # However, if one task hangs, it's bad. 
        # check_prime has a time check inside the loop.
        
        try:
            results = pool.imap_unordered(worker, tasks)
            for p, k in results:
                if time.time() > deadline:
                    break
                if k > best_k:
                    best_k = k
                    best_p = p
        except Exception as e:
            pass
            
    # If we have time, we can try to extend the search for the best_p found
    # or search larger primes.
    # But the prompt says "Maximize k".
    # If we found a good k, we might want to verify it or find a better one.
    
    # For now, return the best found.
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
