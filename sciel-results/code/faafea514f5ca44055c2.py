"""Randomized search with optimized factorization for long distinct-totient runs."""
# EVOLVE-BLOCK-START
import random
import math

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

# Primes up to 10000 for fast trial division
PRIMES = get_primes(10000)

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    x = random.randint(2, n - 1)
    y = x
    d = 1
    f = lambda x: (x * x + 1) % n
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
    if d != n:
        return d
    return pollard_rho(n)

def factorize(n):
    factors = []
    temp_n = n
    for p in PRIMES:
        if p * p > temp_n:
            break
        while temp_n % p == 0:
            factors.append(p)
            temp_n //= p
    if temp_n > 1:
        if is_prime(temp_n):
            factors.append(temp_n)
        else:
            # Use Pollard Rho
            while temp_n > 1:
                d = pollard_rho(temp_n)
                if d == temp_n:
                    continue
                factors.append(d)
                temp_n //= d
    return factors

def phi_from_factors(n, factors):
    if n == 1:
        return 1
    result = n
    unique_factors = set(factors)
    for p in unique_factors:
        result -= result // p
    return result

def construct():
    MAX_N = 10**12
    MAX_K = 200
    best_k = 1
    best_s = 1
    
    # Number of trials
    # Adjust based on performance
    num_trials = 20000
    
    for _ in range(num_trials):
        s = random.randint(1, MAX_N - MAX_K)
        
        phi_vals = []
        for i in range(MAX_K):
            n = s + i
            factors = factorize(n)
            phi_val = phi_from_factors(n, factors)
            phi_vals.append(phi_val)
            
        # Check distinctness
        if len(set(phi_vals)) == MAX_K:
            best_k = MAX_K
            best_s = s
            break
            
        # Find longest run
        seen = set()
        left = 0
        current_max = 0
        for right in range(MAX_K):
            v = phi_vals[right]
            if v in seen:
                while left <= right:
                    seen.discard(phi_vals[left])
                    left += 1
                    if left > right:
                        break
            seen.add(v)
            cur_len = right - left + 1
            if cur_len > current_max:
                current_max = cur_len
        
        if current_max > best_k:
            best_k = current_max
            best_s = s + left

    return {'start': int(best_s), 'length': int(best_k)}
# EVOLVE-BLOCK-END
