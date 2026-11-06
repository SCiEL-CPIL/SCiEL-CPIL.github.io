import math
import time
from math import gcd

_DEADLINE = 145.0
_START = time.time()

def _tau_perp(n):
    """
    Compute the score tau_perp(n): number of adjacent coprime pairs in sorted divisors.
    """
    if n < 2:
        return 0
    
    # Factorize n
    x = n
    factors = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            count = 0
            while x % d == 0:
                x //= d
                count += 1
            factors.append((d, count))
        d += 1 if d == 2 else 2
    
    if x > 1:
        factors.append((x, 1))
        
    # Generate divisors
    divs = [1]
    for p, e in factors:
        new_divs = []
        pk = 1
        for k in range(e + 1):
            for d in divs:
                new_divs.append(d * pk)
            pk *= p
        divs = new_divs
        
    divs.sort()
    
    if len(divs) < 2:
        return 0
        
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def construct():
    global _START
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    max_n = 10**18
    
    # Use a larger pool of primes for the search
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = []
    
    def dfs(idx, max_exp, current_n, current_tau, current_factors):
        if time.time() > deadline - 15:
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n, list(current_factors)))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_n // current_n
        
        # Max exponent allowed by value constraint
        e_max_val = 0
        temp = 1
        while temp * p <= limit_n:
            temp *= p
            e_max_val += 1
            
        # Max exponent allowed by non-increasing constraint
        e_max = min(max_exp, e_max_val)
        
        # Try exponents from 0 to e_max (0 means skip this prime)
        for e in range(0, e_max + 1):
            if e == 0:
                next_n = current_n
                next_tau = current_tau
                next_factors = list(current_factors)
            else:
                next_n = current_n * (p ** e)
                next_tau = current_tau * (e + 1)
                next_factors = list(current_factors) + [(p, e)]
            
            dfs(idx + 1, e, next_n, next_tau, next_factors)
            
            if time.time() > deadline - 15:
                return

    # Start DFS
    dfs(0, 60, 1, 1, [])
    
    if not candidates:
        # Fallback
        n = 1
        for p in primes[:15]:
            n *= p
        return n
    
    # Sort by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = -1
    best_n = candidates[0][1]
    
    eval_deadline = start_time + _DEADLINE - 3.0
    
    # Evaluate top candidates
    for tau, n, factors in candidates:
        if time.time() > eval_deadline:
            break
            
        score = _tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Local search: try replacing primes with nearby alternatives
    if time.time() < eval_deadline - 1.0:
        # Generate neighbor candidates by replacing one prime with another
        evaluated = set()
        for tau, n, factors in candidates[:200]:
            if time.time() > eval_deadline:
                break
            if n in evaluated:
                continue
            evaluated.add(n)
            
            # Try replacing each prime in the factorization
            for i, (p, e) in enumerate(factors):
                # Try nearby primes
                p_idx = primes.index(p) if p in primes else -1
                if p_idx >= 0:
                    for offset in [-2, -1, 1, 2, 3]:
                        new_idx = p_idx + offset
                        if 0 <= new_idx < len(primes):
                            new_p = primes[new_idx]
                            if new_p == p:
                                continue
                            
                            # Build new n
                            new_n = 1
                            for j, (pj, ej) in enumerate(factors):
                                if j == i:
                                    new_n *= new_p ** ej
                                else:
                                    new_n *= pj ** ej
                            
                            if new_n > max_n or new_n in evaluated:
                                continue
                            
                            evaluated.add(new_n)
                            score = _tau_perp(new_n)
                            if score > best_score:
                                best_score = score
                                best_n = new_n
                                if time.time() > eval_deadline:
                                    break
                    if time.time() > eval_deadline:
                        break
                if time.time() > eval_deadline:
                    break
            
            if time.time() > eval_deadline:
                break
    
    return best_n
