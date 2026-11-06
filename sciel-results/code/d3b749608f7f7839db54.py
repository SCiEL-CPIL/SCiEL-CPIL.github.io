import math
from functools import lru_cache

def _get_divisors(num):
    """Generate all divisors of num in unsorted order."""
    divisors = [1]
    i = 2
    while i * i <= num:
        if num % i == 0:
            power = 1
            while num % i == 0:
                num //= i
                power *= i
                for d in list(divisors):
                    divisors.append(d * power)
        i += 1
    if num > 1:
        for d in list(divisors):
            divisors.append(d * num)
    return divisors

def _score_n(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    divs = _get_divisors(n)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes(limit):
    """Generate all primes up to limit using sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    import time
    start_time = time.time()
    time_limit = 110  # leave 10s buffer
    
    # Small primes to consider
    primes = _generate_primes(200)
    
    # Strategy: For squarefree numbers, the score depends on the set of primes.
    # We search over subsets of primes, prioritizing smaller primes but also
    # considering that the product must be <= 10^18.
    
    best_n = 2
    best_score = _score_n(2)
    
    # First, try the primorial of first k primes for various k
    # Primorial of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # = 614889782588491410 ~ 6.15e17, fits in 10^18
    
    # Compute primorials
    primorials = []
    prod = 1
    for p in primes:
        prod *= p
        if prod > 10**18:
            break
        primorials.append((prod, len(primorials)))
    
    # Evaluate primorials
    for n_val, k in primorials:
        s = _score_n(n_val)
        if s > best_score:
            best_score = s
            best_n = n_val
    
    # Now search for better combinations by swapping primes
    # The idea: replace larger primes with smaller ones that aren't already included,
    # or try different subsets of similar size
    
    # Get the set of primes used in the best primorial
    # Let's do a local search: try removing one prime and adding another
    
    # For efficiency, precompute which primes fit
    # Try subsets of size 14-16 from first ~25 primes
    
    # Use a greedy + local search approach
    # Start with the best primorial found, then try swaps
    
    # Determine how many primes are in best_n
    def factorize(n):
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
        return factors
    
    current_factors = factorize(best_n)
    current_set = set(current_factors)
    
    # Local search: try swapping one prime out and one in
    improved = True
    while improved:
        improved = False
        for i, p_out in enumerate(sorted(current_set)):
            for p_in in primes:
                if p_in in current_set:
                    continue
                # Check if p_in * (product of others) <= 10^18
                new_prod = 1
                for p in current_set:
                    if p != p_out:
                        new_prod *= p
                new_prod *= p_in
                if new_prod > 10**18:
                    continue
                new_set = (current_set - {p_out}) | {p_in}
                new_n = 1
                for p in sorted(new_set):
                    new_n *= p
                s = _score_n(new_n)
                if s > best_score:
                    best_score = s
                    best_n = new_n
                    current_set = new_set
                    improved = True
                    break
            if time.time() - start_time > time_limit:
                return best_n
            if improved:
                break
    
    return best_n

# EVOLVE-BLOCK-END
