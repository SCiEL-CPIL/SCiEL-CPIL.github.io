import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000)
    
    def tau_perp_fast(prime_factors):
        """Compute tau_perp for squarefree n with given distinct primes."""
        k = len(prime_factors)
        if k == 0:
            return 0
        # Generate all divisors (subset products)
        divs = [1]
        for p in prime_factors:
            divs.extend([d * p for d in divs])
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initialize with smallest primes until product exceeds 10^18
    initial_primes = []
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        initial_primes.append(p)
        prod *= p
        
    best_n = prod
    best_score = tau_perp_fast(initial_primes)
    best_primes = list(initial_primes)

    current_primes = list(initial_primes)
    current_prod = prod
    current_score = best_score

    # Precompute: for each prime in current set, which primes can replace it
    def get_candidate_primes(p_old, current_prod, current_primes):
        max_p_new = int((10**18 * p_old) / current_prod)
        candidates = []
        for p_new in primes:
            if p_new > max_p_new:
                break
            if p_new == p_old:
                continue
            if p_new in current_primes:
                continue
            candidates.append(p_new)
        return candidates

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            
            best_swap_score = -1
            best_swap_prime = -1
            
            for p_new in primes:
                if p_new * current_prod // p_old > 10**18:
                    break
                if p_new == p_old:
                    continue
                if p_new in current_primes:
                    continue
                
                new_primes = list(current_primes)
                new_primes[idx] = p_new
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_swap_score:
                    best_swap_score = s_new
                    best_swap_prime = p_new
            
            if best_swap_prime != -1 and best_swap_score > current_score:
                current_primes[idx] = best_swap_prime
                current_prod = 1
                for p in current_primes:
                    current_prod *= p
                current_score = best_swap_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)
                    
            if time.time() > deadline - 5.0:
                break

    return best_n
