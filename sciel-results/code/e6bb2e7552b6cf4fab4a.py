import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(300) # Slightly larger pool for swaps

def _tau_perp(primes, exponents):
    """
    Compute tau_perp score from prime factorization.
    """
    # Generate divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    
    divisors.sort()
    if len(divisors) < 2:
        return 0
    
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def _local_search(primes, exponents, deadline, max_n):
    """
    Perform local search on exponents for a fixed set of primes.
    Returns best (score, n, exponents).
    """
    current_primes = primes
    current_exps = exponents[:]
    current_n = _compute_n(current_primes, current_exps)
    current_score = _tau_perp(current_primes, current_exps)
    
    improved = True
    iterations = 0
    max_iterations = 2000 # Cap to prevent infinite loops in case of bugs, though score is bounded
    
    while improved and iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1
        
        # 1. Increase exponent
        for i in range(len(current_primes)):
            if time.time() > deadline:
                return current_score, current_n, current_exps
            p = current_primes[i]
            new_n = current_n * p
            if new_n <= max_n:
                new_exps = current_exps[:]
                new_exps[i] += 1
                score = _tau_perp(current_primes, new_exps)
                if score > current_score:
                    current_exps = new_exps
                    current_n = new_n
                    current_score = score
                    improved = True
                    break # Restart scan with first improvement
                    
        # 2. Decrease exponent
        for i in range(len(current_primes)):
            if time.time() > deadline:
                return current_score, current_n, current_exps
            if current_exps[i] <= 1:
                continue
            p = current_primes[i]
            new_n = current_n // p
            if new_n >= 2:
                new_exps = current_exps[:]
                new_exps[i] -= 1
                score = _tau_perp(current_primes, new_exps)
                if score > current_score:
                    current_exps = new_exps
                    current_n = new_n
                    current_score = score
                    improved = True
                    break

    return current_score, current_n, current_exps

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    max_n = 10**18
    
    best_score = 0
    best_n = 2
    best_exps = [1]
    best_primes = [2]
    
    # Candidate prime sets
    # 1. Primorial 15
    # 2. Primorial 14
    # 3. Primorial 13
    
    # Find max primorial size
    prod = 1
    max_k = 0
    primorials = []
    for i, p in enumerate(_PRIMES):
        if prod * p <= max_n:
            prod *= p
            max_k = i + 1
            primorials.append((i+1, prod))
        else:
            break
            
    # We know max_k is 15.
    # We will generate candidate sets based on the first max_k primes,
    # but allow swapping the largest few primes with larger ones.
    
    candidates = []
    
    # Base candidates: Primorial k for k in [13, 14, 15]
    for k in range(max(1, max_k-2), max_k+1):
        p_set = _PRIMES[:k]
        e_set = [1]*k
        candidates.append((p_set, e_set))
        
        # Swaps for the largest primes in the set
        # Try swapping p[k-1] with larger primes
        for j in range(k, min(k+10, len(_PRIMES))):
            new_p = _PRIMES[j]
            # Check if n stays within limit
            # n_base = product of p_set
            # n_new = n_base * new_p / p_set[-1]
            # Since p_set[-1] is the largest, and new_p > p_set[-1], n increases.
            # We need to check if it fits.
            # We can just compute it in the loop below if we add to candidates.
            # But to avoid adding too many, we check roughly.
            # Actually, let's just add them and let the local search / evaluation handle the bound check?
            # No, _compute_n will handle it, but we should filter out obviously too large ones.
            
            # Let's just add the set and let the local search start.
            # But if the base n is already > max_n, we should skip.
            # For primorial 15, n ~ 6e17.
            # Swapping 47 for 53: n ~ 6.9e17. OK.
            # Swapping 47 for 59: n ~ 7.6e17. OK.
            # Swapping 47 for 61: n ~ 7.9e17. OK.
            # Swapping 47 for 67: n ~ 8.7e17. OK.
            # Swapping 47 for 71: n ~ 9.2e17. OK.
            # Swapping 47 for 73: n ~ 9.5e17. OK.
            # Swapping 47 for 79: n ~ 1.03e18. Too big.
            
            # So we can swap 47 for primes up to 73.
            
            new_p_set = p_set[:-1] + [new_p]
            new_e_set = e_set[:]
            candidates.append((new_p_set, new_e_set))
            
            # Try swapping the two largest
            if k >= 2:
                new_p_set2 = p_set[:-2] + [new_p] + [_PRIMES[j+1] if j+1 < len(_PRIMES) else new_p]
                # This logic is getting messy. Let's just stick to single swaps for the largest prime.
                # And maybe double swaps for the largest two.
                
    # To be safe and systematic, let's generate candidates programmatically:
    # 1. Start with first k primes.
    # 2. Allow the last m primes to be replaced by any larger primes, such that the product stays <= max_n.
    
    # Simpler approach:
    # Iterate through possible "bases" (first k primes).
    # For each base, run local search.
    # Additionally, for the largest base (k=15), try swapping the largest prime p_15 with p_16, p_17, etc.
    
    search_targets = []
    
    # Target 1: P15 (first 15 primes)
    search_targets.append((_PRIMES[:15], [1]*15))
    
    # Target 2: P15 with largest prime swapped
    # 47 -> 53, 59, 61, 67, 71, 73
    for p_new in [53, 59, 61, 67, 71, 73]:
        search_targets.append((_PRIMES[:14] + [p_new], [1]*15))
        
    # Target 3: P14 (first 14 primes)
    search_targets.append((_PRIMES[:14], [1]*14))
    
    # Target 4: P14 with largest prime swapped (43 -> 47, 53, ...)
    for p_new in [47, 53, 59]:
         search_targets.append((_PRIMES[:13] + [p_new], [1]*14))
         
    # Target 5: P13
    search_targets.append((_PRIMES[:13], [1]*13))

    # Run local search on each target
    for primes, exps in search_targets:
        if time.time() > deadline:
            break
            
        # Check if base n is valid
        n_base = _compute_n(primes, exps)
        if n_base > max_n:
            continue
            
        score, n, final_exps = _local_search(primes, exps, deadline, max_n)
        
        if score > best_score:
            best_score = score
            best_n = n
            best_exps = final_exps
            best_primes = primes

    # Final polish: Try to improve best_n by swapping primes in the best set found
    # This is a second pass.
    if best_score > 0 and time.time() < deadline:
        # We have best_primes and best_exps.
        # Try swapping each prime in best_primes with a larger prime from _PRIMES.
        # This is expensive, so we only do it for the best candidate.
        
        current_primes = best_primes
        current_exps = best_exps
        current_n = best_n
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_primes)):
                p_old = current_primes[i]
                # Find larger primes
                # We need to ensure the new set has distinct primes.
                # Since we start with distinct primes, and we swap p_old with p_new > p_old,
                # we must ensure p_new is not already in the set.
                # Since p_new > p_old, and p_old is in the set, and the set is sorted?
                # We should keep the set sorted for consistency.
                
                # Try swapping p_old with p_new
                # n_new = n * p_new / p_old
                # If n_new <= max_n:
                #   new_set = (current_primes without p_old) + [p_new]
                #   Check if p_new is in current_primes. If so, skip.
                #   Run local search on new_set with same exponents (adjusted? no, exponents are per prime).
                #   Actually, if we swap primes, the exponents should stay with the "position" or the "prime"?
                #   Usually, we associate exponents with primes.
                #   If we swap p_old with p_new, we take the exponent of p_old and assign it to p_new?
                #   Or do we keep exponents fixed to the "slot"?
                #   The local search optimizes exponents for the specific primes.
                #   So if we change the primes, we should restart local search with exps=[1]*k?
                #   Or use the current exponents as a starting point?
                #   Using current exponents is a good heuristic.
                
                for p_new in _PRIMES:
                    if p_new <= p_old:
                        continue
                    if p_new in current_primes:
                        continue
                    n_new = current_n * p_new // p_old
                    if n_new > max_n:
                        break # Since _PRIMES is sorted, larger p_new will be even bigger
                    if n_new < 2:
                        continue
                        
                    # Construct new prime list
                    new_primes = [p for p in current_primes if p != p_old]
                    new_primes.append(p_new)
                    new_primes.sort()
                    
                    # Construct new exponent list
                    # We need to map exponents to the new primes.
                    # This is tricky if we just swap.
                    # Let's just restart local search with exps=[1]*k for simplicity and robustness.
                    # The local search will find the best exponents for this new set.
                    
                    new_exps = [1] * len(new_primes)
                    
                    s, n_final, e_final = _local_search(new_primes, new_exps, deadline, max_n)
                    
                    if s > current_score:
                        current_score = s
                        current_n = n_final
                        current_exps = e_final
                        current_primes = new_primes
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        break # Found improvement, restart outer loop

    return best_n
