import math
import time
from itertools import combinations

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number (product of distinct primes)."""
    divisors = [1]
    for p in primes:
        new_divs = [d * p for d in divisors]
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_squarefree(primes):
    """Count adjacent coprime pairs among divisors of a squarefree number."""
    divs = _divisors_of_squarefree(primes)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return -1, n
    return _tau_perp_squarefree(prime_list), n

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate a list of primes up to a reasonable limit
    # The 15th prime in a 15-prime set is at most ~77 if others are small.
    # But if we swap a small prime for a large one, the others must be very small.
    # Let's generate primes up to 1000 to be safe for intermediate swaps.
    primes_list = []
    for i in range(2, 1000):
        if _is_prime(i):
            primes_list.append(i)
            
    all_primes = primes_list
    
    best_score = -1
    best_n = 2
    best_primes = [2]
    
    def try_candidate(prime_list):
        nonlocal best_score, best_n, best_primes
        if time.time() > deadline:
            return False
        s, n = _score_primes(prime_list)
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
            best_primes = list(prime_list)
        return True
    
    # Initialize with primorials
    for k in range(2, 16):
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > 10**18:
            break
        try_candidate(primes_k)
        
    # Local Search / Hill Climbing
    # We will iterate until no improvement is found or time runs out.
    # To avoid infinite loops or slow convergence, we limit iterations.
    max_iterations = 100
    no_improvement_count = 0
    
    current_primes = list(best_primes)
    current_score = best_score
    current_n = best_n
    
    while time.time() < deadline and no_improvement_count < 5:
        improved = False
        current_primes.sort()
        
        # Try single swaps
        for i in range(len(current_primes)):
            p_to_remove = current_primes[i]
            rest_product = current_n // p_to_remove
            
            # Find candidate primes to swap in
            # The new prime q must satisfy:
            # 1. q is not in current_primes (except p_to_remove which is removed)
            # 2. rest_product * q <= 10**18
            # 3. q is prime
            
            max_q = 10**18 // rest_product
            if max_q < 2:
                continue
                
            # We iterate through primes_list to find valid q
            # Optimization: We can break early if primes_list[j] > max_q
            for q in all_primes:
                if q > max_q:
                    break
                if q == p_to_remove:
                    continue
                if q in current_primes:
                    continue
                    
                new_primes = [p for j, p in enumerate(current_primes) if j != i]
                new_primes.append(q)
                new_primes.sort()
                
                s, n = _score_primes(new_primes)
                if n <= 10**18:
                    if s > current_score:
                        current_score = s
                        current_n = n
                        current_primes = new_primes
                        improved = True
                        # Restart swap loop to explore from new point
                        break 
            if improved:
                break
                
        if not improved:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_primes = list(current_primes)
                
    return best_n
