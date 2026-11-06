"""Constructs an integer n <= 10^18 maximizing tau_perp(n)."""

import time
from math import gcd, log2
from itertools import combinations, product

def _tau_perp_fast(n, primes, exps):
    """Compute tau_perp for n given its prime factorization."""
    k = len(primes)
    if k == 0:
        return 0
    # Generate all divisors
    divs = [1]
    for i, (p, e) in enumerate(zip(primes, exps)):
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    
    if all(e == 1 for e in exps):
        # Square-free: use bitmask
        divisor_mask = {}
        for d in divs:
            mask = 0
            for j, p in enumerate(primes):
                if d % p == 0:
                    mask |= (1 << j)
            divisor_mask[d] = mask
        count = 0
        for i in range(len(divs) - 1):
            if divisor_mask[divs[i]] & divisor_mask[divs[i+1]] == 0:
                count += 1
        return count
    else:
        count = 0
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i + 1]) == 1:
                count += 1
        return count

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    best_n = 2
    best_score = 0
    
    # First 15 primes
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def make_n(primes, exps):
        n = 1
        for p, e in zip(primes, exps):
            n *= (p ** e)
        return n
    
    def try_candidate(n, primes, exps):
        nonlocal best_n, best_score
        if n < 2 or n > 10**18:
            return
        if time.time() > deadline - 3:
            return
        s = _tau_perp_fast(n, primes, exps)
        if s > best_score:
            best_score = s
            best_n = n
    
    # === Strategy: Small-prime-heavy exponents + square-free tail ===
    # n = 2^a * 3^b * 5^c * 7^d * (product of subset of remaining primes)
    # Search over (a,b,c,d) and subset of {11,13,17,...,47}
    
    # Phase 1: Enumerate exponent vectors for small primes
    # Max exponents: 2^a <= 10^18 -> a <= 59, but combined with others much less
    # Use beam search over exponent combinations
    
    # Precompute: for each subset of tail primes, what's the product?
    tail_primes = all_primes[4:]  # 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    
    # Generate all subsets of tail primes that fit in 10^18
    # But we'll combine with small-prime exponents, so be conservative
    # Precompute tail products and their sizes
    tail_subsets = []
    n_tail = len(tail_primes)
    for mask in range(1 << n_tail):
        prod = 1
        for i in range(n_tail):
            if mask & (1 << i):
                prod *= tail_primes[i]
                if prod > 10**18:
                    break
        if prod <= 10**18:
            tail_subsets.append((mask, prod))
    
    # Sort tail subsets by product (smaller first, more budget for exponents)
    tail_subsets.sort(key=lambda x: x[1])
    
    # For each tail subset, search over exponent allocations for 2,3,5,7
    # Use a beam approach: try many exponent combos, keep best
    
    # Max exponents for small primes (conservative bounds)
    max_a = 60  # 2^60 ~ 10^18
    max_b = 38  # 3^38 ~ 10^18
    max_c = 26  # 5^26 ~ 10^18
    max_d = 21  # 7^21 ~ 10^18
    
    # For efficiency, we'll iterate over tail subsets and for each,
    # try a grid of exponent combinations
    
    # First: try the case with no tail (pure small-prime exponents)
    # and increasing tail sizes
    
    evaluated = 0
    
    for mask, tail_prod in tail_subsets:
        if time.time() > deadline - 10:
            break
        
        # Budget for small primes: 10^18 / tail_prod
        budget = 10**18 // tail_prod
        if budget < 2:
            continue
        
        # Determine max exponents given budget
        ba = 0
        temp = 1
        while temp * 2 <= budget:
            temp *= 2
            ba += 1
        bb = 0
        temp = 1
        while temp * 3 <= budget:
            temp *= 3
            bb += 1
        bc = 0
        temp = 1
        while temp * 5 <= budget:
            temp *= 5
            bc += 1
        bd = 0
        temp = 1
        while temp * 7 <= budget:
            temp *= 7
            bd += 1
        
        # Try a grid of exponent combinations
        # To keep it tractable, use a coarser grid
        # Step sizes depend on budget
        step_a = max(1, ba // 15)
        step_b = max(1, bb // 12)
        step_c = max(1, bc // 10)
        step_d = max(1, bd // 8)
        
        # Build the full prime list and exponent list
        tail_mask_primes = [tail_primes[i] for i in range(n_tail) if mask & (1 << i)]
        
        # Try combinations
        for a in range(0, ba + 1, step_a):
            if time.time() > deadline - 10:
                break
            rem_a = budget // (2 ** a)
            if rem_a < 1:
                break
            for b in range(0, bb + 1, step_b):
                rem_b = rem_a // (3 ** b)
                if rem_b < 1:
                    break
                for c in range(0, bc + 1, step_c):
                    rem_c = rem_b // (5 ** c)
                    if rem_c < 1:
                        break
                    for d in range(0, bd + 1, step_d):
                        rem_d = rem_c // (7 ** d)
                        if rem_d < 1:
                            break
                        
                        # Check total
                        n_try = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * tail_prod
                        if n_try > 10**18 or n_try < 2:
                            continue
                        
                        # Build primes and exps
                        primes_list = [2, 3, 5, 7] + tail_mask_primes
                        exps_list = [a, b, c, d] + [1] * len(tail_mask_primes)
                        
                        # Filter out zero exponents
                        filtered_primes = []
                        filtered_exps = []
                        for p, e in zip(primes_list, exps_list):
                            if e > 0:
                                filtered_primes.append(p)
                                filtered_exps.append(e)
                        
                        if len(filtered_primes) < 1:
                            continue
                        
                        try_candidate(n_try, filtered_primes, filtered_exps)
                        evaluated += 1
                        if evaluated % 500 == 0 and time.time() > deadline - 10:
                            break
    
    # Phase 2: Local refinement around best found
    if time.time() < deadline - 15:
        # Factorize best_n
        temp = best_n
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        primes_list = sorted(factors.keys())
        exps_list = [factors[p] for p in primes_list]
        
        # Try incrementing/decrementing exponents
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, p in enumerate(primes_list):
                # Try increasing exponent
                if exps_list[i] < 60:
                    new_exps = exps_list.copy()
                    new_exps[i] += 1
                    n_try = make_n(primes_list, new_exps)
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try, primes_list, new_exps)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
                            exps_list = new_exps
                            improved = True
                            break
                # Try decreasing exponent (if > 1)
                if exps_list[i] > 1:
                    new_exps = exps_list.copy()
                    new_exps[i] -= 1
                    n_try = make_n(primes_list, new_exps)
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try, primes_list, new_exps)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
                            exps_list = new_exps
                            improved = True
                            break
        
        # Try adding a new prime
        for p in all_primes:
            if p not in primes_list and time.time() < deadline - 5:
                new_primes = primes_list + [p]
                new_primes.sort()
                new_exps = [exps_list[primes_list.index(p2)] if p2 in primes_list else 1 for p2 in new_primes]
                n_try = make_n(new_primes, new_exps)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try, new_primes, new_exps)
                    if s > best_score:
                        best_score = s
                        best_n = n_try
                        break
        
        # Try removing a prime
        for i in range(len(primes_list)):
            if time.time() > deadline - 5:
                break
            new_primes = primes_list[:i] + primes_list[i+1:]
            new_exps = exps_list[:i] + exps_list[i+1:]
            if len(new_primes) < 1:
                continue
            n_try = make_n(new_primes, new_exps)
            if 2 <= n_try <= 10**18:
                s = _tau_perp_fast(n_try, new_primes, new_exps)
                if s > best_score:
                    best_score = s
                    best_n = n_try
                    break
    
    return best_n
