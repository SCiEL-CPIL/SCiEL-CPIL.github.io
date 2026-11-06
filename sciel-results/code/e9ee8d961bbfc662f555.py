"""
Combined approach: greedy + local search with prime-factor-based coprimality checks.
Uses the structural idea of the current program (greedy + local search with deadline guard)
combined with the refinement machinery of inspiration 3 (prime factor precomputation for fast
coprimality checks) and inspiration 2 (swap-based local search with early termination).
"""
import math
import time
from math import gcd

_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_exhausted():
    return _elapsed() > TIME_LIMIT

def _score(chosen, scores):
    """Compute sum 1/(512 - a) as float."""
    s = 0.0
    for a in chosen:
        s += scores[a]
    return s

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 that maximize sum 1/(512 - a).
    
    Strategy:
    1. Precompute prime factors for all numbers 1..511.
    2. Greedy initialization: pick numbers in descending order of 1/(512-a), maintaining coprime constraint.
    3. Local search: try to swap elements to improve the score.
    4. Use a deadline guard to ensure we return the best result found before time runs out.
    """
    # Precompute scores
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512.0 - a)
    
    # Sieve of Eratosthenes for primes up to 512
    limit = 512
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Pre-compute prime factors for all numbers 1..511
    def _get_prime_factors(n, primes):
        """Get unique prime factors of n."""
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        return factors
    
    prime_factors = [set() for _ in range(limit)]
    for i in range(1, limit):
        prime_factors[i] = _get_prime_factors(i, primes)
    
    # Greedy Initialization
    # We want to pick numbers with large 1/(512-a), i.e., large a.
    # Check if a is coprime to all chosen.
    # This is equivalent to: prime_factors[a] is disjoint from union of prime_factors[chosen].
    
    chosen = []
    chosen_prime_factors = set()
    
    for a in range(511, 0, -1):
        if not prime_factors[a]:  # 1 has no prime factors, gcd(1, x) = 1 always
            chosen.append(a)
            # 1 doesn't add any prime factors to the forbidden set
        else:
            # Check disjointness
            if not prime_factors[a].intersection(chosen_prime_factors):
                chosen.append(a)
                chosen_prime_factors.update(prime_factors[a])
                
    best_score = _score(chosen, scores)
    
    # Local Search: try swapping each chosen element with an unchosen one
    # to improve the score while maintaining pairwise coprimality.
    
    improved = True
    while improved and not _time_exhausted():
        improved = False
        chosen_set = set(chosen)
        
        # For each position, try replacing with a candidate not in chosen
        for i in range(len(chosen)):
            if _time_exhausted():
                break
            
            current = chosen[i]
            current_score = scores[current]
            
            # Determine required prime factors (from all other chosen elements)
            req_factors = set()
            for j in range(len(chosen)):
                if j != i:
                    req_factors.update(prime_factors[chosen[j]])
            
            # Find the best replacement
            best_new = -1
            best_new_score = 0.0
            
            # Iterate candidates in descending order of score (descending a)
            # We can stop early if we find a valid one that is better
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if a == current:
                    continue
                
                # Check if a is coprime to all other chosen
                if prime_factors[a] and prime_factors[a].intersection(req_factors):
                    continue
                
                # Check if a is better
                new_score = scores[a]
                if new_score > current_score:
                    best_new = a
                    best_new_score = new_score
                    break  # Found the best candidate for this slot
            
            if best_new != -1:
                # Perform swap
                chosen[i] = best_new
                chosen_set.remove(current)
                chosen_set.add(best_new)
                best_score = best_score - current_score + best_new_score
                improved = True
                # Break out of the inner loop to restart the outer loop
                break
    
    return sorted(chosen)
