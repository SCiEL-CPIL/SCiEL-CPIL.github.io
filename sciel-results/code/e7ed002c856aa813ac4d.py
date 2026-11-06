import math
import time
from math import gcd
from itertools import product

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(1000)

def _tau_perp(n):
    """Compute number of adjacent divisor pairs with gcd 1."""
    if n < 2:
        return 0
    # Generate all divisors
    divisors = [1]
    d = 2
    while d * d <= n:
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
        d += 1
    divisors.append(n)
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp_fast(best_n, [(2,1),(3,1),(5,1),(7,1),(11,1),(13,1),(17,1),(19,1)])
    
    # Strategy 1: Try squarefree numbers with first k primes
    # Primorial of first k primes
    primes = _PRIMES
    for k in range(2, 20):
        if time.time() > deadline - 5:
            break
        # Product of first k primes
        prod = 1
        factorization = []
        valid = True
        for i in range(k):
            if prod * primes[i] > LIMIT:
                valid = False
                break
            prod *= primes[i]
            factorization.append((primes[i], 1))
        if not valid or k < 2:
            continue
        score = _tau_perp_fast(prod, factorization)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Strategy 2: Try replacing some primes with larger ones to fit more primes
    # Or try non-squarefree: p^2 for small p might help?
    # Actually, for squarefree, more primes = more divisors = potentially more coprime adjacent pairs
    # But the arrangement matters.
    
    # Strategy 3: Try numbers of form p1 * p2 * ... * pk where we can choose which primes
    # The first k primes give the smallest product, allowing maximum k.
    
    # Strategy 4: Try adding a squared small prime to a primorial
    # e.g., 2^2 * 3 * 5 * 7 * ... might give more divisors
    base_primes = primes[:15]  # First 15 primes fit
    for trial in range(200):
        if time.time() > deadline - 3:
            break
        # Try: take first (k-1) primes, replace one with a slightly larger prime, or add exponent
        # Actually let's try: for each position, try exponent 2 on a small prime
        for k in range(8, 16):
            if time.time() > deadline - 3:
                break
            for exp_pos in range(min(k, 4)):  # Try squaring first few primes
                prod = 1
                factorization = []
                valid = True
                for i in range(k):
                    exp = 2 if i == exp_pos else 1
                    if prod * (primes[i] ** exp) > LIMIT:
                        valid = False
                        break
                    prod *= primes[i] ** exp
                    factorization.append((primes[i], exp))
                if not valid:
                    continue
                # Only evaluate if number of divisors is reasonable
                num_divs = 1
                for _, e in factorization:
                    num_divs *= (e + 1)
                if num_divs > 200000:  # Too many to enumerate quickly
                    continue
                score = _tau_perp_fast(prod, factorization)
                if score > best_score:
                    best_score = score
                    best_n = prod
    
    return best_n

def construct():
    return _search()
# EVOLVE-BLOCK-END
