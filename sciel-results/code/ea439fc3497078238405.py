import time
import math
import random

_score_cache = {}

def _tau_perp_squarefree(primes):
    """
    Calculate tau_perp for a square-free number given its list of prime factors.
    For square-free n with k primes, divisors are all 2^k subsets.
    We generate divisors, sort them, and count adjacent coprime pairs.
    """
    key = tuple(sorted(primes))
    if key in _score_cache:
        return _score_cache[key]
    
    k = len(primes)
    # Generate divisors using bitmasks
    divs = []
    for mask in range(1 << k):
        val = 1
        for i in range(k):
            if mask & (1 << i):
                val *= primes[i]
        divs.append(val)
    
    divs.sort()
    
    score = 0
    prev = divs[0]
    for curr in divs[1:]:
        if math.gcd(prev, curr) == 1:
            score += 1
        prev = curr
    
    _score_cache[key] = score
    return score

def _is_prime(n):
    """Check if n is prime using trial division up to sqrt(n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n == 3:
        return True
    if n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes_up_to(limit, count):
    """Generate the first `count` primes up to `limit`."""
    primes = []
    candidate = 2
    while len(primes) < count and candidate <= limit:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate a larger set of candidate primes
    # We need primes that could replace small ones while keeping product <= 10^18
    # The product of 15 primes should be <= 10^18
    # Start with first 15 primes as base
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    base_product = 1
    for p in base_primes:
        base_product *= p
    
    # Generate candidate primes up to ~2000 for swaps
    # We want enough primes to explore the space
    max_prime_candidate = 5000
    candidate_primes = _generate_primes_up_to(max_prime_candidate, 700)
    
    # Remove the base primes from candidates (they're already in the set)
    base_set = set(base_primes)
    candidate_primes = [p for p in candidate_primes if p not in base_set]
    
    best_n = base_product
    best_score = _tau_perp_squarefree(base_primes)
    current_set = base_primes[:]
    current_product = base_product
    
    # Local search with single swaps
    improved = True
    passes = 0
    while improved and time.time() < DEADLINE:
        improved = False
        passes += 1
        
        # Try swapping each prime in current_set with a candidate prime
        for idx, p_old in enumerate(current_set):
            if time.time() > DEADLINE - 10:
                break
            
            # Find candidate replacement primes
            # We need product / p_old * p_new <= LIMIT
            max_p_new = int(LIMIT / current_product * p_old)
            
            for p_new in candidate_primes:
                if p_new > max_p_new:
                    break
                if p_new in current_set:
                    continue
                
                new_product = current_product // p_old * p_new
                if new_product > LIMIT:
                    continue
                
                new_set = current_set.copy()
                new_set[idx] = p_new
                new_set.sort()
                
                score_new = _tau_perp_squarefree(new_set)
                if score_new > best_score:
                    best_score = score_new
                    best_n = new_product
                    current_set = new_set
                    current_product = new_product
                    improved = True
                    break
            
            if improved:
                break
        
        # If no single swap improves, try double swaps
        if not improved and time.time() < DEADLINE - 20:
            for idx1 in range(len(current_set)):
                for idx2 in range(idx1 + 1, len(current_set)):
                    if time.time() > DEADLINE - 10:
                        break
                    p_old1 = current_set[idx1]
                    p_old2 = current_set[idx2]
                    max_p_new = int(LIMIT / current_product * p_old1 * p_old2)
                    
                    for p_new1 in candidate_primes:
                        if p_new1 > max_p_new:
                            break
                        if p_new1 in current_set:
                            continue
                        for p_new2 in candidate_primes:
                            if p_new2 <= p_new1:
                                continue
                            if p_new2 > max_p_new:
                                break
                            if p_new2 in current_set:
                                continue
                            
                            new_product = current_product // p_old1 // p_old2 * p_new1 * p_new2
                            if new_product > LIMIT:
                                continue
                            
                            new_set = current_set.copy()
                            new_set[idx1] = p_new1
                            new_set[idx2] = p_new2
                            new_set.sort()
                            
                            score_new = _tau_perp_squarefree(new_set)
                            if score_new > best_score:
                                best_score = score_new
                                best_n = new_product
                                current_set = new_set
                                current_product = new_product
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if time.time() > DEADLINE - 15:
            break

    # Final check
    if best_n == 0 or best_n < 2:
        best_n = 1
        for p in base_primes:
            best_n *= p
    
    assert 2 <= best_n <= LIMIT
    return best_n
