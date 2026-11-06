import time
import math
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    factors = {}
    tmp = n
    d = 2
    while d * d <= tmp:
        while tmp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            tmp //= d
        d += 1
    if tmp > 1:
        factors[tmp] = factors.get(tmp, 0) + 1
    
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    return divs

def _score(n):
    """Compute tau_perp(n): number of adjacent divisor pairs with gcd 1."""
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _factorize(n):
    """Return list of (prime, exponent) pairs."""
    factors = []
    tmp = n
    d = 2
    while d * d <= tmp:
        if tmp % d == 0:
            e = 0
            while tmp % d == 0:
                tmp //= d
                e += 1
            factors.append((d, e))
        d += 1
    if tmp > 1:
        factors.append((tmp, 1))
    return factors

def _value_from_factors(factors):
    """Compute n from list of (prime, exponent)."""
    n = 1
    for p, e in factors:
        n *= p ** e
    return n

def _try_local_search(start_factors, deadline, best_n, best_score):
    """Local search over exponent changes and prime swaps."""
    factors = list(start_factors)
    current_n = _value_from_factors(factors)
    current_score = _score(current_n)
    
    if current_score > best_score:
        best_score = current_score
        best_n = current_n
    
    # Primes to consider (first 20 primes)
    primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(factors)):
            if time.time() > deadline - 5:
                break
            p, e = factors[i]
            # Try increasing exponent
            new_factors = list(factors)
            new_factors[i] = (p, e + 1)
            new_n = _value_from_factors(new_factors)
            if new_n <= 10**18:
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
            # Try decreasing exponent
            if e > 1:
                new_factors = list(factors)
                new_factors[i] = (p, e - 1)
                new_n = _value_from_factors(new_factors)
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
            # Try swapping prime with another
            for j, (p2, e2) in enumerate(factors):
                if j == i:
                    continue
                # Swap primes i and j (keep exponents)
                new_factors = list(factors)
                new_factors[i] = (p2, e)
                new_factors[j] = (p, e2)
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score(new_n)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
        # Try adding a new prime factor
        if time.time() < deadline - 5:
            for p in primes_pool:
                if any(p == pf for pf, _ in factors):
                    continue
                new_factors = factors + [(p, 1)]
                new_n = _value_from_factors(new_factors)
                if new_n <= 10**18:
                    s = _score(new_n)
                    if s > current_score:
                        factors = new_factors
                        current_n = new_n
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
        # Try removing a prime factor
        if time.time() < deadline - 5 and len(factors) > 2:
            for i in range(len(factors)):
                new_factors = factors[:i] + factors[i+1:]
                new_n = _value_from_factors(new_factors)
                s = _score(new_n)
                if s > current_score:
                    factors = new_factors
                    current_n = new_n
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
    
    return best_n, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    max_val = 10**18
    
    candidates = []
    
    # DFS to generate candidates with non-increasing exponents
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > deadline - 10:
            return
            
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
            
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > deadline - 10:
                return

    # Start DFS
    dfs(0, 60, 1, 1)
    
    # Also add the square-free number with maximum primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    k = len([p for p in primes if sf_n % p == 0])
    candidates.append((2**k, sf_n))
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    # Sort candidates by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_n = candidates[0][1]
    best_score = -1
    
    # Evaluate top candidates
    eval_count = min(len(candidates), 500)
    
    for tau, n in candidates[:eval_count]:
        if time.time() - start_time > deadline - 5:
            break
        score = _score(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Local search from the best found
    if time.time() < deadline - 5:
        start_factors = _factorize(best_n)
        best_n, best_score = _try_local_search(start_factors, deadline, best_n, best_score)
        
    # Local search from primorial P15 specifically
    if time.time() < deadline - 5:
        primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        start_factors = [(p, 1) for p in primes_15]
        best_n, best_score = _try_local_search(start_factors, deadline, best_n, best_score)

    return best_n

# EVOLVE-BLOCK-END
