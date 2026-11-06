"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses efficient divisor generation with bitset-based coprime checking
and a broader search over prime exponent configurations.
"""
import math
import time
import itertools


def _tau_perp_from_factorization(primes, exponents):
    """Compute tau_perp(n) given its prime factorization.
    
    Args:
        primes: list of distinct primes
        exponents: list of corresponding exponents
    
    Returns:
        Number of adjacent coprime pairs in the sorted divisor list.
    """
    k = len(primes)
    
    # For each divisor, track a bitmask of which prime indices divide it
    divisors = [(1, 0)]  # (value, bitmask)
    
    for i, (p, e) in enumerate(zip(primes, exponents)):
        new_divs = []
        for d, mask in divisors:
            val = d
            for j in range(e + 1):
                if j == 0:
                    new_mask = mask
                else:
                    new_mask = mask | (1 << i)
                new_divs.append((val, new_mask))
                val *= p
        divisors = new_divs
    
    divisors.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divisors) - 1):
        if divisors[i][1] & divisors[i+1][1] == 0:
            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate primes up to ~200 for broader search
    def _primes_up_to(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    all_primes = _primes_up_to(300)
    
    best_n = 1
    best_score = 0
    
    def try_candidate(primes_list, exponents_list):
        nonlocal best_n, best_score
        n = 1
        for p, e in zip(primes_list, exponents_list):
            n *= p ** e
        if 2 <= n <= 10**18:
            score = _tau_perp_from_factorization(primes_list, exponents_list)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Phase 1: Square-free with 15 primes - try various prime sets
    # Start with smallest 15 primes
    p15 = all_primes[:15]
    try_candidate(p15, [1] * 15)
    
    # Phase 2: Systematic search over 15-prime square-free sets
    # Strategy: fix first k primes, search for remaining 15-k primes
    for num_fixed in range(10, 14):
        if time.time() > deadline:
            break
        fixed_primes = all_primes[:num_fixed]
        fixed_prod = 1
        for p in fixed_primes:
            fixed_prod *= p
        
        remaining_needed = 15 - num_fixed
        min_prime = all_primes[num_fixed]
        
        # Greedy: try to pick the smallest possible primes for remaining
        # Then try swapping larger ones
        def _search_remaining(start_idx, count_needed, current_primes, current_prod):
            nonlocal best_n, best_score
            if time.time() > deadline:
                return
            if count_needed == 0:
                if len(current_primes) == 15:
                    try_candidate(current_primes, [1] * 15)
                return
            if len(current_primes) + (len(all_primes) - start_idx) < count_needed:
                return
            
            for i in range(start_idx, len(all_primes)):
                p = all_primes[i]
                new_prod = current_prod * p
                if new_prod > 10**18:
                    break
                _search_remaining(i + 1, count_needed - 1, current_primes + [p], new_prod)
        
        _search_remaining(num_fixed, remaining_needed, fixed_primes, fixed_prod)
    
    # Phase 3: Higher exponent configurations on small primes
    # Key insight: 2^a * 3^b * 5^c * ... with a,b,c > 1 creates more divisors
    # and different adjacency patterns
    if time.time() > deadline:
        return best_n
    
    # For small number of distinct primes (up to 10), do exponent search
    # with non-increasing exponents to minimize n
    for num_primes in range(5, 11):
        if time.time() > deadline:
            break
        base_primes = all_primes[:num_primes]
        
        def _search_exponents(idx, current_exps, current_prod):
            nonlocal best_n, best_score
            if time.time() > deadline:
                return
            if idx == num_primes:
                if current_prod >= 2 and current_prod <= 10**18:
                    used_primes = []
                    used_exps = []
                    for i in range(num_primes):
                        if current_exps[i] >= 1:
                            used_primes.append(base_primes[i])
                            used_exps.append(current_exps[i])
                    if len(used_primes) >= 2:
                        try_candidate(used_primes, used_exps)
                return
            
            p = base_primes[idx]
            
            # Max exponent for this prime given current product
            max_e = 0
            temp = current_prod
            while temp * p <= 10**18:
                temp *= p
                max_e += 1
            
            # Exponents must be non-increasing
            if idx == 0:
                max_e_allowed = max_e
            else:
                max_e_allowed = min(max_e, current_exps[idx-1])
            
            for e in range(max_e_allowed + 1):
                new_prod = current_prod * (p ** e)
                if new_prod > 10**18:
                    break
                _search_exponents(idx + 1, current_exps + [e], new_prod)
        
        _search_exponents(0, [0], 1)
    
    # Phase 4: 14 primes with one prime squared
    if time.time() > deadline:
        return best_n
    
    for sq_prime in all_primes[:8]:
        if time.time() > deadline:
            break
        sq_prod = sq_prime ** 2
        
        # Find 13 other primes (not including sq_prime) that minimize product
        other_primes = [p for p in all_primes if p != sq_prime]
        
        # Try consecutive sets of 13 from other_primes
        for start in range(0, min(10, len(other_primes) - 13)):
            if time.time() > deadline:
                break
            candidate = other_primes[start:start+13]
            prod = sq_prod
            for p in candidate:
                prod *= p
            if prod <= 10**18:
                primes_list = candidate + [sq_prime]
                primes_list.sort()
                exponents_list = [2 if p == sq_prime else 1 for p in primes_list]
                try_candidate(primes_list, exponents_list)
    
    # Phase 5: 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1 in all_primes[:6]:
        for sq2 in all_primes[:6]:
            if sq1 >= sq2:
                continue
            if time.time() > deadline:
                break
            sq_prod = sq1 ** 2 * sq2 ** 2
            
            other_primes = [p for p in all_primes if p != sq1 and p != sq2]
            
            for start in range(0, min(8, len(other_primes) - 11)):
                if time.time() > deadline:
                    break
                candidate = other_primes[start:start+11]
                prod = sq_prod
                for p in candidate:
                    prod *= p
                if prod <= 10**18:
                    primes_list = candidate + [sq1, sq2]
                    primes_list.sort()
                    exponents_list = [2 if p in (sq1, sq2) else 1 for p in primes_list]
                    try_candidate(primes_list, exponents_list)
    
    # Phase 6: 12 primes with three squared
    if time.time() > deadline:
        return best_n
    
    for sq1 in all_primes[:5]:
        for sq2 in all_primes[:5]:
            for sq3 in all_primes[:5]:
                if not (sq1 < sq2 < sq3):
                    continue
                if time.time() > deadline:
                    break
                sq_prod = sq1 ** 2 * sq2 ** 2 * sq3 ** 2
                
                other_primes = [p for p in all_primes if p not in (sq1, sq2, sq3)]
                
                for start in range(0, min(6, len(other_primes) - 9)):
                    if time.time() > deadline:
                        break
                    candidate = other_primes[start:start+9]
                    prod = sq_prod
                    for p in candidate:
                        prod *= p
                    if prod <= 10**18:
                        primes_list = candidate + [sq1, sq2, sq3]
                        primes_list.sort()
                        exponents_list = [2 if p in (sq1, sq2, sq3) else 1 for p in primes_list]
                        try_candidate(primes_list, exponents_list)
    
    return best_n
