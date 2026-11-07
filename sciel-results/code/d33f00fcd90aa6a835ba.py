"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline with 15s safety margin for 90s limit
    
    N = 256
    
    # Find all primes up to N
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # For each prime, compute the largest prime power <= N
    def max_prime_power(p, limit):
        result = 1
        while result * p <= limit:
            result *= p
        return result
    
    prime_powers = {p: max_prime_power(p, N) for p in primes}
    
    # Get all prime powers (these are pairwise coprime)
    all_prime_powers = [prime_powers[p] for p in primes]
    
    # Baseline: all prime powers + 1
    baseline_set = [1] + all_prime_powers
    baseline_sum = sum(baseline_set)
    
    # Now try to improve by finding beneficial merges
    # A merge replaces a set of prime powers with a composite number that is
    # larger than the sum of the replaced prime powers.
    
    # For a composite to be beneficial, it must be a product of powers of
    # distinct primes, and the composite must exceed the sum of the individual
    # prime powers it replaces.
    
    # Let's enumerate all possible composites that could be beneficial.
    # A composite n = p1^a1 * p2^a2 * ... * pk^ak where p1, ..., pk are distinct primes.
    # The alternative is sum of max_prime_power(pi) for each pi.
    # n is beneficial if n > sum(max_prime_power(pi) for i in 1..k).
    
    # Since N=256 is small, we can enumerate all numbers from 2 to 256 and check
    # which ones are "beneficial" composites.
    
    # First, let's identify which numbers are prime powers (already in our set).
    prime_power_set = set(all_prime_powers)
    
    # For each composite number, determine if it's beneficial.
    # A composite is beneficial if it's not a prime power AND
    # composite > sum of max_prime_power(p) for each prime p dividing composite.
    
    beneficial_composites = []
    for n in range(2, N + 1):
        if n in prime_power_set:
            continue
        # Find prime factors of n
        m = n
        prime_factors = set()
        for p in primes:
            if p * p > m:
                break
            while m % p == 0:
                prime_factors.add(p)
                m //= p
        if m > 1:
            prime_factors.add(m)
        
        # Calculate the sum of max prime powers for these primes
        alt_sum = sum(prime_powers[p] for p in prime_factors)
        
        # Check if n is beneficial
        if n > alt_sum:
            # Also check that n is not already in our set (it shouldn't be since it's composite)
            # And that all prime factors are in our prime list (they should be)
            beneficial_composites.append((n, frozenset(prime_factors), alt_sum))
    
    # Sort by benefit (n - alt_sum) descending
    beneficial_composites.sort(key=lambda x: x[0] - x[2], reverse=True)
    
    # Now we need to select a subset of these beneficial composites such that
    # no prime is used in more than one composite.
    # This is a weighted set packing problem.
    
    # Since the number of beneficial composites is small, we can try to solve this
    # with a simple greedy approach or branch and bound.
    
    # Let's use a greedy approach: sort by benefit, and add if no conflict.
    # Then try to improve with local search.
    
    # Start with all prime powers
    selected_composites = []
    used_primes = set()
    
    for composite, prime_factors, alt_sum in beneficial_composites:
        if not any(p in used_primes for p in prime_factors):
            selected_composites.append((composite, prime_factors, alt_sum))
            used_primes.update(prime_factors)
    
    # Build the result set
    result_set = {1}
    # Add prime powers for primes not used in any composite
    for p in primes:
        if p not in used_primes:
            result_set.add(prime_powers[p])
    # Add the composites
    for composite, prime_factors, alt_sum in selected_composites:
        result_set.add(composite)
    
    # Convert to list and verify
    result_list = list(result_set)
    
    # Verify pairwise coprimality
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            if math.gcd(result_list[i], result_list[j]) != 1:
                # This shouldn't happen, but if it does, fall back to baseline
                return baseline_set
    
    # Verify all elements are in 1..256
    for x in result_list:
        if x < 1 or x > N:
            return baseline_set
    
    # Check if this is better than baseline
    current_sum = sum(result_list)
    
    # Try to improve further with local search
    # We can try swapping composites or adding/removing elements.
    
    # For now, let's just return the greedy result if it's better
    if current_sum > baseline_sum:
        return result_list
    else:
        return baseline_set
# EVOLVE-BLOCK-END
