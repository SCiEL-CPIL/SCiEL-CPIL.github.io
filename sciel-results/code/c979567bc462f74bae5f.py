import time
import math

def _tau_perp_squarefree(primes):
    """
    Calculate tau_perp for a square-free number given its list of prime factors.
    """
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
    return score

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate primes up to 500 for a larger candidate pool
    all_primes = _sieve_primes(500)
    
    # The maximum number of distinct prime factors for n <= 10^18 is 15.
    # Product of first 15 primes: 2*3*...*47 = 614889782588491410 ~ 6.15 * 10^17
    # So we are looking for sets of 15 distinct primes.
    
    best_n = 0
    best_score = -1
    best_set = []
    
    # Strategy: Start with first 15 primes, then do local search with expanded prime pool
    initial_set = all_primes[:15]
    n_std = 1
    for p in initial_set:
        n_std *= p
    if n_std <= LIMIT:
        score_std = _tau_perp_squarefree(initial_set)
        best_score = score_std
        best_n = n_std
        best_set = initial_set[:]
    
    # Local search: try swapping each prime in the set with any other prime from the pool
    # that keeps the product <= LIMIT
    if best_n > 0:
        current_set = best_set[:]
        current_score = best_score
        
        # Multiple rounds of local search to escape local optima
        for round_num in range(20):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Try all possible single swaps
            for idx in range(len(current_set)):
                if time.time() > DEADLINE - 5:
                    break
                p_old = current_set[idx]
                
                # Try swapping with each prime in the pool
                for p_new in all_primes:
                    if p_new == p_old:
                        continue
                    if p_new in current_set:
                        continue
                    
                    new_product = (best_n // p_old) * p_new
                    if new_product > LIMIT:
                        continue
                    
                    new_set = current_set.copy()
                    new_set[idx] = p_new
                    new_set.sort()
                    
                    score_new = _tau_perp_squarefree(new_set)
                    if score_new > best_score:
                        best_score = score_new
                        best_n = new_product
                        best_set = new_set[:]
                        current_set = new_set
                        current_score = score_new
                        improved = True
                        break  # Restart with the improved set
                if improved:
                    break
            
            # If no improvement in this round, try a random perturbation to escape local optimum
            if not improved:
                if time.time() > DEADLINE - 5:
                    break
                # Try a random swap to escape local optimum
                import random
                for _ in range(10):
                    if time.time() > DEADLINE - 5:
                        break
                    idx1 = random.randint(0, len(current_set) - 1)
                    p_old = current_set[idx1]
                    # Pick a random prime to swap with
                    candidates = [p for p in all_primes if p != p_old and p not in current_set]
                    if not candidates:
                        continue
                    p_new = random.choice(candidates)
                    new_product = (best_n // p_old) * p_new
                    if new_product > LIMIT:
                        continue
                    
                    new_set = current_set.copy()
                    new_set[idx1] = p_new
                    new_set.sort()
                    
                    score_new = _tau_perp_squarefree(new_set)
                    # Accept even if not better, to escape local optima (simulated annealing style)
                    # But keep track of best
                    if score_new > best_score:
                        best_score = score_new
                        best_n = new_product
                        best_set = new_set[:]
                        current_set = new_set
                        current_score = score_new
                        break
                    elif random.random() < 0.3:  # 30% chance to accept worse solution
                        current_set = new_set
                        current_score = score_new
    
    if best_n == 0:
        # Fallback to first 15 primes
        best_n = 1
        for p in all_primes[:15]:
            best_n *= p
            
    assert 2 <= best_n <= LIMIT
    return best_n
