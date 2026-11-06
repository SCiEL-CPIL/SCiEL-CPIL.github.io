"""Constructs an integer n <= 10^18 maximizing tau_perp(n)."""

import time
from math import gcd

def _tau_perp_fast(n):
    """Compute tau_perp(n): count of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    primes = list(factors.keys())
    exps = [factors[p] for p in primes]
    k = len(primes)
    
    # Generate divisors
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    
    divs.sort()
    
    # For square-free numbers, use bitmask gcd check
    if all(e == 1 for e in exps):
        divisor_mask = {}
        for i, d in enumerate(divs):
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 2
    best_score = 0
    
    # Prime lists
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    primes_13 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    primes_12 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def make_n(primes, exps):
        n = 1
        for p, e in zip(primes, exps):
            n *= (p ** e)
        return n
    
    def try_candidate(n):
        nonlocal best_n, best_score
        if n < 2 or n > 10**18:
            return
        if time.time() > deadline - 3:
            return
        s = _tau_perp_fast(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    # === k=15: only P15 itself (no room for extra exponents) ===
    P15 = 1
    for p in primes_15:
        P15 *= p
    if P15 <= 10**18:
        try_candidate(P15)
    
    # === k=14: drop 47, optimize exponents on first few primes ===
    if time.time() < deadline - 10:
        prod_14_base = 1
        for p in primes_14:
            prod_14_base *= p
        # prod_14_base = 2*3*5*7*11*13*17*19*23*29*31*37*41*43
        # budget for extra: 10^18 / prod_14_base
        budget_14 = 10**18 // prod_14_base
        
        # Search over extra exponents for 2,3,5,7,11 (indices 0-4)
        # n = prod_14_base * 2^a * 3^b * 5^c * 7^d * 11^e
        # where a,b,c,d,e >= 0 and 2^a*3^b*5^c*7^d*11^e <= budget_14
        
        # Use beam search
        # State: (a, b, c, d, e) -> value = 2^a*3^b*5^c*7^d*11^e
        # Start from (0,0,0,0,0)
        
        candidates = [(0, 0, 0, 0, 0, 1)]  # (a,b,c,d,e, mult)
        seen = {(0, 0, 0, 0, 0)}
        
        max_beam_steps = 50
        for step in range(max_beam_steps):
            if time.time() > deadline - 8:
                break
            
            new_candidates = []
            for a, b, c, d, e, mult in candidates:
                # Try incrementing each exponent
                for i in range(5):
                    pks = [2, 3, 5, 7, 11]
                    pk = pks[i]
                    if i == 0:
                        new_a, new_b, new_c, new_d, new_e = a+1, b, c, d, e
                    elif i == 1:
                        new_a, new_b, new_c, new_d, new_e = a, b+1, c, d, e
                    elif i == 2:
                        new_a, new_b, new_c, new_d, new_e = a, b, c+1, d, e
                    elif i == 3:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d+1, e
                    else:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d, e+1
                    
                    new_mult = mult * pk
                    if new_mult > budget_14:
                        continue
                    
                    key = (new_a, new_b, new_c, new_d, new_e)
                    if key not in seen:
                        seen.add(key)
                        new_candidates.append((new_a, new_b, new_c, new_d, new_e, new_mult))
            
            if not new_candidates:
                break
            
            # Score top candidates
            # Limit scoring to avoid timeout
            limit = 20
            scored = []
            for cand in new_candidates[:limit]:
                a, b, c, d, e, mult = cand
                exps = [1+a, 1+b, 1+c, 1+d, 1+e] + [1]*9  # 14 primes total
                n_try = make_n(primes_14, exps)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    scored.append((s, n_try, cand))
                if time.time() > deadline - 5:
                    break
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_try, cand in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_try
            
            # Keep top for next round
            if scored:
                candidates = [(x[2][0], x[2][1], x[2][2], x[2][3], x[2][4], x[2][5]) for x in scored[:10]]
            else:
                break
    
    # === k=13: drop 43,47, more budget ===
    if time.time() < deadline - 10:
        prod_13_base = 1
        for p in primes_13:
            prod_13_base *= p
        budget_13 = 10**18 // prod_13_base
        
        # n = prod_13_base * 2^a * 3^b * 5^c * 7^d * 11^e
        # Search
        candidates = [(0, 0, 0, 0, 0, 1)]
        seen = {(0, 0, 0, 0, 0)}
        
        max_beam_steps = 50
        for step in range(max_beam_steps):
            if time.time() > deadline - 8:
                break
            
            new_candidates = []
            for a, b, c, d, e, mult in candidates:
                for i in range(5):
                    pks = [2, 3, 5, 7, 11]
                    pk = pks[i]
                    if i == 0:
                        new_a, new_b, new_c, new_d, new_e = a+1, b, c, d, e
                    elif i == 1:
                        new_a, new_b, new_c, new_d, new_e = a, b+1, c, d, e
                    elif i == 2:
                        new_a, new_b, new_c, new_d, new_e = a, b, c+1, d, e
                    elif i == 3:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d+1, e
                    else:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d, e+1
                    
                    new_mult = mult * pk
                    if new_mult > budget_13:
                        continue
                    
                    key = (new_a, new_b, new_c, new_d, new_e)
                    if key not in seen:
                        seen.add(key)
                        new_candidates.append((new_a, new_b, new_c, new_d, new_e, new_mult))
            
            if not new_candidates:
                break
            
            limit = 20
            scored = []
            for cand in new_candidates[:limit]:
                a, b, c, d, e, mult = cand
                exps = [1+a, 1+b, 1+c, 1+d, 1+e] + [1]*8  # 13 primes total
                n_try = make_n(primes_13, exps)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    scored.append((s, n_try, cand))
                if time.time() > deadline - 5:
                    break
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_try, cand in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_try
            
            if scored:
                candidates = [(x[2][0], x[2][1], x[2][2], x[2][3], x[2][4], x[2][5]) for x in scored[:10]]
            else:
                break
    
    # === k=12: drop 41,43,47, even more budget ===
    if time.time() < deadline - 10:
        prod_12_base = 1
        for p in primes_12:
            prod_12_base *= p
        budget_12 = 10**18 // prod_12_base
        
        candidates = [(0, 0, 0, 0, 0, 1)]
        seen = {(0, 0, 0, 0, 0)}
        
        max_beam_steps = 50
        for step in range(max_beam_steps):
            if time.time() > deadline - 8:
                break
            
            new_candidates = []
            for a, b, c, d, e, mult in candidates:
                for i in range(5):
                    pks = [2, 3, 5, 7, 11]
                    pk = pks[i]
                    if i == 0:
                        new_a, new_b, new_c, new_d, new_e = a+1, b, c, d, e
                    elif i == 1:
                        new_a, new_b, new_c, new_d, new_e = a, b+1, c, d, e
                    elif i == 2:
                        new_a, new_b, new_c, new_d, new_e = a, b, c+1, d, e
                    elif i == 3:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d+1, e
                    else:
                        new_a, new_b, new_c, new_d, new_e = a, b, c, d, e+1
                    
                    new_mult = mult * pk
                    if new_mult > budget_12:
                        continue
                    
                    key = (new_a, new_b, new_c, new_d, new_e)
                    if key not in seen:
                        seen.add(key)
                        new_candidates.append((new_a, new_b, new_c, new_d, new_e, new_mult))
            
            if not new_candidates:
                break
            
            limit = 20
            scored = []
            for cand in new_candidates[:limit]:
                a, b, c, d, e, mult = cand
                exps = [1+a, 1+b, 1+c, 1+d, 1+e] + [1]*7  # 12 primes total
                n_try = make_n(primes_12, exps)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    scored.append((s, n_try, cand))
                if time.time() > deadline - 5:
                    break
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_try, cand in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_try
            
            if scored:
                candidates = [(x[2][0], x[2][1], x[2][2], x[2][3], x[2][4], x[2][5]) for x in scored[:10]]
            else:
                break
    
    return best_n
