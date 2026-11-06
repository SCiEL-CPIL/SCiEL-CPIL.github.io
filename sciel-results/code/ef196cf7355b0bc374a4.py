"""
Constructs an integer n <= 10^18 maximizing tau_perp(n).

Approach: Model the sorted divisor sequence and use greedy selection of
primes to maximize the density of coprime adjacent pairs. We search
over combinations of primes (and possibly prime powers) to find the
n that maximizes the count of adjacent coprime divisor pairs.
"""

import math

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    divs = [1]
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            new_divs = []
            for d in divs:
                mult = 1
                for e in range(exp + 1):
                    new_divs.append(d * mult)
                    mult *= p
            divs = new_divs
        p += 1 if p == 2 else 2
    if temp > 1:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * temp)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _primes_up_to(limit):
    """Generate primes up to limit using simple sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _optimize_exponents(primes, limit):
    """
    Greedy search for exponents of the given primes to maximize _tau_perp
    under the constraint that the product <= limit.
    """
    k = len(primes)
    exponents = [1] * k
    current_n = 1
    for i in range(k):
        current_n *= primes[i]
    
    if current_n > limit:
        return None, None

    best_score = _tau_perp(current_n)
    best_exponents = exponents[:]
    best_n = current_n

    improved = True
    while improved:
        improved = False
        for i in range(k):
            # Try increasing exponent of primes[i] by 1
            new_exponents = exponents[:]
            new_exponents[i] += 1
            new_n = current_n * primes[i]
            if new_n > limit:
                continue
            
            new_score = _tau_perp(new_n)
            if new_score > best_score:
                best_score = new_score
                best_exponents = new_exponents
                best_n = new_n
                exponents = new_exponents
                current_n = new_n
                improved = True
                break
            
    return best_n, best_score

def construct():
    all_primes = _primes_up_to(100)
    best_n = 2
    best_score = 0
    
    # Try sets of the first k primes
    for k in range(2, 16):
        if k > len(all_primes):
            break
        primes = all_primes[:k]
        # Check if product of first k primes is within limit
        prod = 1
        for p in primes:
            prod *= p
        if prod > 10**18:
            break
            
        # Optimize exponents for this set
        n, score = _optimize_exponents(primes, 10**18)
        if n is not None and score > best_score:
            best_score = score
            best_n = n
            
    return best_n
