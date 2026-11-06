"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses a more efficient divisor generation and a broader search over
prime exponent configurations, including:
  - 15-prime square-free (primorial)
  - 15-prime with some primes swapped for larger ones
  - 14-prime with one prime squared
  - 13-prime with two primes squared
  - 14-prime with one prime cubed (if product allows)
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
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        p_powers = [1]
        for _ in range(e):
            p_powers.append(p_powers[-1] * p)
        for d in divisors:
            for pp in p_powers:
                new_divs.append(d * pp)
        divisors = new_divs
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First 25 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
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
    
    # Phase 1: Square-free with 15 primes (the primorial of first 15 primes)
    p15 = base_primes[:15]
    try_candidate(p15, [1] * 15)
    
    # Phase 2: Try swapping the largest 1-3 primes with larger ones
    for num_swaps in range(1, 4):
        if time.time() > deadline:
            break
        fixed_primes = base_primes[:15 - num_swaps]
        fixed_prod = 1
        for p in fixed_primes:
            fixed_prod *= p
        
        max_allowed = 10**18 // fixed_prod
        # Try combinations of larger primes
        larger_start = 15 - num_swaps
        larger_primes = base_primes[larger_start:]
        
        if num_swaps == 1:
            for p in larger_primes:
                if fixed_prod * p <= 10**18:
                    try_candidate(fixed_primes + [p], [1] * 15)
                else:
                    break
        elif num_swaps == 2:
            for p1_idx in range(len(larger_primes)):
                p1 = larger_primes[p1_idx]
                if fixed_prod * p1 > 10**18:
                    break
                for p2_idx in range(p1_idx + 1, len(larger_primes)):
                    p2 = larger_primes[p2_idx]
                    if fixed_prod * p1 * p2 > 10**18:
                        break
                    try_candidate(fixed_primes + [p1, p2], [1] * 15)
        elif num_swaps == 3:
            for p1_idx in range(len(larger_primes)):
                p1 = larger_primes[p1_idx]
                if fixed_prod * p1 > 10**18:
                    break
                for p2_idx in range(p1_idx + 1, len(larger_primes)):
                    p2 = larger_primes[p2_idx]
                    if fixed_prod * p1 * p2 > 10**18:
                        break
                    for p3_idx in range(p2_idx + 1, len(larger_primes)):
                        p3 = larger_primes[p3_idx]
                        if fixed_prod * p1 * p2 * p3 > 10**18:
                            break
                        try_candidate(fixed_primes + [p1, p2, p3], [1] * 15)
    
    # Phase 3: 14 primes with one prime squared
    if time.time() > deadline:
        return best_n
    
    for sq_idx in range(6):  # Try squaring primes at index 0-5
        if time.time() > deadline:
            break
        sq_prime = base_primes[sq_idx]
        sq_prod = sq_prime ** 2
        
        # Try different sets of 13 other primes
        # Start with consecutive primes, then try variations
        for offset in range(0, 5):
            if time.time() > deadline:
                break
            candidate_primes = []
            prod = sq_prod
            # Choose 13 primes, skipping sq_idx
            count = 0
            for i in range(25):
                if i == sq_idx:
                    continue
                p = base_primes[i]
                if prod * p > 10**18:
                    break
                candidate_primes.append(p)
                prod *= p
                count += 1
                if count == 13:
                    break
            
            if count == 13:
                # Ensure sq_prime is in the list (it might not be if we skipped it)
                if sq_prime not in candidate_primes:
                    # Find where to insert it
                    candidate_primes.append(sq_prime)
                    candidate_primes.sort()
                    # But we need to adjust: remove the largest prime to keep product <= 10^18
                    if len(candidate_primes) == 14:
                        # Try removing different primes
                        for rem_idx in range(len(candidate_primes) - 1, -1, -1):
                            if candidate_primes[rem_idx] != sq_prime:
                                test_primes = candidate_primes[:rem_idx] + candidate_primes[rem_idx+1:]
                                test_exps = [1] * 14
                                # Set exponent of sq_prime to 2
                                sq_pos = test_primes.index(sq_prime)
                                test_exps[sq_pos] = 2
                                try_candidate(test_primes, test_exps)
                                break
                else:
                    # sq_prime is already in candidate_primes
                    # We need 13 other primes + sq_prime = 14 total, but we only have 13 in candidate_primes
                    # This means we need to find one more prime
                    # Let's just use the first 14 primes with sq_prime squared
                    pass
            
            # Simpler approach: use consecutive primes with one squared
            for start_idx in range(0, 5):
                if time.time() > deadline:
                    break
                primes_list = []
                exponents_list = []
                prod = 1
                for i in range(start_idx, start_idx + 14):
                    if i >= len(base_primes):
                        break
                    p = base_primes[i]
                    e = 2 if p == sq_prime else 1
                    if prod * p ** e > 10**18:
                        break
                    primes_list.append(p)
                    exponents_list.append(e)
                    prod *= p ** e
                
                if len(primes_list) == 14:
                    try_candidate(primes_list, exponents_list)
    
    # Phase 4: 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1_idx, sq2_idx in itertools.combinations(range(6), 2):
        if time.time() > deadline:
            break
        sq1 = base_primes[sq1_idx]
        sq2 = base_primes[sq2_idx]
        sq_prod = sq1 ** 2 * sq2 ** 2
        
        # Try different sets of 11 other primes
        for start_idx in range(0, 5):
            if time.time() > deadline:
                break
            primes_list = []
            exponents_list = []
            prod = 1
            for i in range(start_idx, start_idx + 13):
                if i >= len(base_primes):
                    break
                p = base_primes[i]
                if p == sq1:
                    e = 2
                elif p == sq2:
                    e = 2
                else:
                    e = 1
                if prod * p ** e > 10**18:
                    break
                primes_list.append(p)
                exponents_list.append(e)
                prod *= p ** e
            
            if len(primes_list) == 13:
                try_candidate(primes_list, exponents_list)
    
    # Phase 5: 14 primes with one prime cubed (if product allows)
    if time.time() > deadline:
        return best_n
    
    for cube_idx in range(4):  # Try cubing primes at index 0-3
        if time.time() > deadline:
            break
        cube_prime = base_primes[cube_idx]
        cube_prod = cube_prime ** 3
        
        for start_idx in range(0, 4):
            if time.time() > deadline:
                break
            primes_list = []
            exponents_list = []
            prod = 1
            for i in range(start_idx, start_idx + 14):
                if i >= len(base_primes):
                    break
                p = base_primes[i]
                e = 3 if p == cube_prime else 1
                if prod * p ** e > 10**18:
                    break
                primes_list.append(p)
                exponents_list.append(e)
                prod *= p ** e
            
            if len(primes_list) == 14:
                try_candidate(primes_list, exponents_list)
    
    # Phase 6: Try some non-consecutive prime sets with 15 primes
    if time.time() > deadline:
        return best_n
    
    # Try replacing one of the middle primes with a larger one
    for swap_idx in range(3, 12):  # Swap primes at index 3-11
        if time.time() > deadline:
            break
        base_set = base_primes[:15]
        removed = base_set[swap_idx]
        base_prod = 1
        for p in base_set:
            base_prod *= p
        
        # Find primes larger than removed that fit
        for i in range(swap_idx + 1, len(base_primes)):
            p = base_primes[i]
            new_prod = base_prod * p // removed
            if new_prod > 10**18:
                break
            new_set = base_set[:swap_idx] + [p] + base_set[swap_idx+1:]
            try_candidate(new_set, [1] * 15)
    
    return best_n
