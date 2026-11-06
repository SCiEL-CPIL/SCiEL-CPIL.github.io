import math
import time
import functools

@functools.lru_cache(maxsize=10000)
def _score_from_factorization_cached(prime_tuple, exp_tuple):
    """
    Calculate tau_perp given the prime factorization.
    Uses caching for repeated evaluations.
    """
    primes = list(prime_tuple)
    exponents = list(exp_tuple)
    
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
            
    return count

def _score_from_factorization(primes, exponents):
    """
    Calculate tau_perp given the prime factorization.
    primes: list of primes
    exponents: list of exponents
    """
    return _score_from_factorization_cached(tuple(primes), tuple(exponents))

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 110  # Safety margin
    
    max_val = 10**18
    
    # First 20 primes for broader search
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Seed fallback
    
    # 1. Evaluate Maximal Primorial P15 (known to be strong)
    p15_val = 1
    for p in primes[:15]:
        p15_val *= p
    if p15_val <= max_val:
        score_p15 = _score_from_factorization(primes[:15], [1]*15)
        if score_p15 > best_score:
            best_score = score_p15
            best_n = p15_val
            
    # 2. Evaluate variations with fewer primes but higher exponents
    candidates = []
    
    def add_candidate(p_list, exps):
        val = 1
        for p, e in zip(p_list, exps):
            val *= (p ** e)
        if val <= max_val and val >= 2:
            candidates.append((p_list, exps, val))
            
    # Try removing 1-3 largest primes from P15 and increasing exponents
    for num_remove in range(0, 4):
        base = primes[:15 - num_remove]
        base_len = len(base)
        if base_len < 1:
            continue
        # Try increasing exponents of first 3 primes
        for i in range(min(3, base_len)):
            for e in range(2, 8):
                exps = [1] * base_len
                exps[i] = e
                add_candidate(base, exps)
        # Two prime exponent increases
        for i in range(min(3, base_len)):
            for j in range(i+1, min(4, base_len)):
                for ei in range(2, 5):
                    for ej in range(2, 5):
                        exps = [1] * base_len
                        exps[i] = ei
                        exps[j] = ej
                        add_candidate(base, exps)
        
    # Evaluate all candidates
    for p_list, e_list, val in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        score = _score_from_factorization(p_list, e_list)
        if score > best_score:
            best_score = score
            best_n = val
            
    # 3. DFS with pruning - explore non-increasing exponent sequences
    def dfs(idx, max_exp, current_n, current_primes, current_exps):
        nonlocal best_score, best_n
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if len(current_primes) > 0:
            score = _score_from_factorization(current_primes, current_exps)
            if score > best_score:
                best_score = score
                best_n = current_n

        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try e=1 first (square-free path - most promising for coprime pairs)
        if 1 <= max_exp:
            new_n = current_n * p
            if new_n <= max_val:
                dfs(idx + 1, 1, new_n, current_primes + [p], current_exps + [1])
                
        # Then try other exponents > 1 (only if we have room)
        for e in range(min(max_exp, 10), 1, -1):
            if time.time() - start_time > TIME_LIMIT:
                return
            new_n = current_n * (p ** e)
            if new_n > max_val:
                continue
            dfs(idx + 1, e, new_n, current_primes + [p], current_exps + [e])
            
        # Skip this prime
        dfs(idx + 1, max_exp, current_n, current_primes, current_exps)

    # Start DFS with moderate max_exp to explore non-square-free structures
    dfs(0, 20, 1, [], [])
    
    return best_n

# EVOLVE-BLOCK-END
