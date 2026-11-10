import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000000)

    # 2. Optimized tau_perp with bitmask-based gcd check
    def tau_perp_fast(prime_factors):
        k = len(prime_factors)
        if k == 0:
            return 0
            
        # Build list of (value, bitmask) for all divisors
        divs = [(1, 0)]
        
        for i, p in enumerate(prime_factors):
            bit = 1 << i
            current_len = len(divs)
            for j in range(current_len):
                val, mask = divs[j]
                divs.append((val * p, mask | bit))
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count adjacent coprime pairs using bitmask AND
        score = 0
        for i in range(len(divs) - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    # 3. Initialize with product of smallest 15 primes
    max_k = 15
    current_primes = list(primes[:max_k])
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    best_score = tau_perp_fast(current_primes)
    current_score = best_score
    current_prod = current_n

    # Helper to check if a swap is valid
    def is_valid_swap(p_out, p_in, prod, limit=10**18):
        return p_in * prod <= limit * p_out

    # 4. Local search
    improved = True
    no_improvement_count = 0
    max_no_improvement = 5
    
    while improved and time.time() < deadline - 15.0:
        improved = False
        current_prime_set = set(current_primes)
        
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                return best_n
                
            p_old = current_primes[idx]
            
            valid_candidates = []
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new * current_prod <= 10**18 * p_old:
                    valid_candidates.append(p_new)
                else:
                    pass
            
            test_candidates = []
            if len(valid_candidates) > 0:
                test_candidates.extend(valid_candidates[:20])
                if len(valid_candidates) > 40:
                    test_candidates.extend(valid_candidates[-20:])
            
            best_candidate = None
            best_candidate_score = current_score - 1
            
            for p_new in test_candidates:
                if p_new in current_prime_set:
                    continue
                if p_new * current_prod > 10**18 * p_old:
                    continue
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None and best_candidate_score > current_score:
                current_primes[idx] = best_candidate
                current_prod = (current_prod // p_old) * best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                improved = True
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                if time.time() < deadline - 30.0:
                    perturbed = current_primes.copy()
                    for _ in range(3):
                        idx = random.randint(0, len(perturbed)-1)
                        p_old = perturbed[idx]
                        valids = [p for p in primes if p not in set(perturbed) and p * (current_prod // p_old) <= 10**18]
                        if valids:
                            p_new = random.choice(valids)
                            perturbed[idx] = p_new
                    perturbed_score = tau_perp_fast(perturbed)
                    if perturbed_score > best_score:
                        best_score = perturbed_score
                        best_n = 1
                        for p in perturbed:
                            best_n *= p
                        current_primes = perturbed
                        current_score = perturbed_score
                        current_prod = best_n
                        improved = True
                        no_improvement_count = 0
                else:
                    break

    return best_n
