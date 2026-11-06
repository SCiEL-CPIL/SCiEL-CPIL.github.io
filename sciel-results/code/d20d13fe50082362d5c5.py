import time
import math
from math import gcd, isqrt
from itertools import combinations
from functools import lru_cache

# EVOLVE-BLOCK-START

# Precompute primes up to a reasonable limit for fast factorization
def _sieve(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

_SMALL_PRIMES = _sieve(100000)
_SMALL_PRIMES_SET = set(_SMALL_PRIMES)

def _is_prime(n):
    """Deterministic Miller-Rabin for n < 2^64."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def _factorize(n):
    """Return sorted list of (prime, exponent) pairs."""
    factors = []
    tmp = n
    for p in _SMALL_PRIMES:
        if p * p > tmp:
            break
        if tmp % p == 0:
            e = 0
            while tmp % p == 0:
                tmp //= p
                e += 1
            factors.append((p, e))
    if tmp > 1:
        if _is_prime(tmp):
            factors.append((tmp, 1))
        else:
            d = 2
            while d * d <= tmp:
                if tmp % d == 0:
                    e = 0
                    while tmp % d == 0:
                        tmp //= d
                        e += 1
                    factors.append((d, e))
                    break
                d += 1 if d == 2 else 2
            if tmp > 1:
                factors.append((tmp, 1))
    factors.sort()
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from factor list, sorted."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pk = 1
        for k in range(e + 1):
            for d in divs:
                new_divs.append(d * pk)
            pk *= p
        divs = new_divs
    divs.sort()
    return divs

def _score_from_factors(factors):
    """Compute tau_perp from factor list."""
    divs = _divisors_from_factors(factors)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _value_from_factors(factors):
    """Compute n from list of (prime, exponent)."""
    n = 1
    for p, e in factors:
        n *= p ** e
    return n

def _try_local_search(factors, deadline, best_n, best_score, max_iter=5000):
    """Local search over exponent changes, prime swaps, adds, removes."""
    factors = [list(f) for f in factors]
    current_n = _value_from_factors(factors)
    if current_n > 10**18:
        return best_n, best_score
    current_score = _score_from_factors(factors)
    
    if current_score > best_score:
        best_score = current_score
        best_n = current_n
    
    primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    iters = 0
    improved = True
    while improved and time.time() < deadline - 3 and iters < max_iter:
        improved = False
        iters += 1
        current_n = _value_from_factors(factors)
        current_score = _score_from_factors(factors)
        
        # Try each prime: increase/decrease exponent
        for i in range(len(factors)):
            if time.time() > deadline - 3:
                return best_n, best_score
            p, e = factors[i]
            # Increase
            new_factors = [list(f) for f in factors]
            new_factors[i][1] = e + 1
            new_n = _value_from_factors(new_factors)
            if new_n <= 10**18:
                s = _score_from_factors(new_factors)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_n = new_n
            # Decrease
            if e > 1:
                new_factors = [list(f) for f in factors]
                new_factors[i][1] = e - 1
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score_from_factors(new_factors)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_n = new_n
        
        # Try swapping primes between positions
        for i in range(len(factors)):
            if time.time() > deadline - 3:
                return best_n, best_score
            for j in range(i + 1, len(factors)):
                p1, e1 = factors[i]
                p2, e2 = factors[j]
                if p1 == p2:
                    continue
                new_factors = [list(f) for f in factors]
                new_factors[i] = [p2, e1]
                new_factors[j] = [p1, e2]
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score_from_factors(new_factors)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_n = new_n
        
        # Try adding a new prime
        if len(factors) < 20:
            for p in primes_pool:
                if any(f[0] == p for f in factors):
                    continue
                new_factors = [list(f) for f in factors] + [[p, 1]]
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score_from_factors(new_factors)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_n = new_n
        
        # Try removing a prime (only if we have many)
        if len(factors) > 6:
            for i in range(len(factors)):
                new_factors = [list(f) for f in factors[:i]] + [list(f) for f in factors[i+1:]]
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score_from_factors(new_factors)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_n = new_n
    
    return best_n, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    best_n = 2
    best_score = 0
    
    # Strategy 1: Primorials with varying numbers of primes
    for num_primes in [16, 15, 14, 13, 12, 11, 10, 9, 8]:
        if time.time() > deadline - 5:
            break
        primes = _SMALL_PRIMES[:num_primes]
        factors = [[p, 1] for p in primes]
        n = _value_from_factors(factors)
        if n <= 10**18:
            s = _score_from_factors(factors)
            if s > best_score:
                best_score = s
                best_n = n
    
    # Strategy 2: Local search from best primorial
    if time.time() < deadline - 10:
        best_pf = None
        for num_primes in [16, 15, 14, 13, 12, 11, 10, 9, 8]:
            primes = _SMALL_PRIMES[:num_primes]
            factors = [[p, 1] for p in primes]
            n = _value_from_factors(factors)
            if n <= 10**18:
                s = _score_from_factors(factors)
                if best_pf is None or s > best_score:
                    best_score = s
                    best_n = n
                    best_pf = [list(f) for f in factors]
        if best_pf:
            best_n, best_score = _try_local_search(best_pf, deadline, best_n, best_score)
    
    # Strategy 3: Try structures with some exponents > 1
    if time.time() < deadline - 10:
        patterns = [
            [(2,2),(3,2),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1),(31,1),(37,1)],
            [(2,3),(3,2),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1),(31,1)],
            [(2,2),(3,2),(5,2),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1),(29,1)],
            [(2,3),(3,3),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1),(23,1)],
            [(2,4),(3,2),(5,2),(7,1),(11,1),(13,1),(17,1),(19,1)],
            [(2,3),(3,3),(5,2),(7,2),(11,1),(13,1),(17,1)],
            [(2,4),(3,4),(5,2),(7,2),(11,1),(13,1)],
            [(2,5),(3,3),(5,3),(7,2),(11,1)],
            [(2,4),(3,4),(5,3),(7,2),(11,2)],
            [(2,5),(3,4),(5,3),(7,3),(11,1)],
        ]
        for pat in patterns:
            if time.time() > deadline - 10:
                break
            factors = [list(f) for f in pat]
            n = _value_from_factors(factors)
            if n <= 10**18:
                s = _score_from_factors(factors)
                if s > best_score:
                    best_score = s
                    best_n = n
                if time.time() < deadline - 15:
                    best_n, best_score = _try_local_search(factors, deadline, best_n, best_score)
    
    # Strategy 4: Random starting points
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            num_primes = random.randint(8, 16)
            primes = random.sample(_SMALL_PRIMES[:30], num_primes)
            primes.sort()
            exponents = [1] * num_primes
            for _ in range(random.randint(0, 3)):
                idx = random.randint(0, num_primes - 1)
                exponents[idx] = random.randint(2, 4)
            factors = [[primes[i], exponents[i]] for i in range(num_primes)]
            n = _value_from_factors(factors)
            if n <= 10**18:
                s = _score_from_factors(factors)
                if s > best_score:
                    best_score = s
                    best_n = n
                if time.time() < deadline - 15:
                    best_n, best_score = _try_local_search(factors, deadline, best_n, best_score)
    
    if best_n < 2:
        best_n = 2
    
    return best_n

# EVOLVE-BLOCK-END
