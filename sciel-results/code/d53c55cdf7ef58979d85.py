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
    # Since n is generated from small primes, we can trial divide by small primes
    # or store the factorization during DFS. 
    # For robustness and simplicity in this combined approach, we will factorize n 
    # using trial division up to sqrt(n) or known small primes.
    # Given n <= 10^18, trial division up to 10^6 is not enough if n has a large prime factor.
    # However, our DFS only uses small primes (first 20). 
    # We can pass the factorization from DFS to avoid refactoring.
    # But _tau_perp is a standalone function in the current program. 
    # Let's make it take factors if possible, or just factorize.
    # Factorization of numbers <= 10^18 with small prime factors is fast.
    
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
        d += 1 if d == 2 else 2 # Optimize: 2, then odd numbers
    
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
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    candidates = []
    
    def dfs(idx, max_exp, current_n, current_tau, current_factors):
        if time.time() > deadline - 10: # Stop generating 10s before deadline
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n, current_factors))
            
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
        
        # Try exponents from 1 to e_max
        # We iterate e from 1 upwards? Or downwards?
        # Standard HCN generation: e from max_exp down to 1.
        # This ensures we find high exponent combinations for small primes early.
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            next_factors = current_factors + [(p, e)]
            
            dfs(idx + 1, e, next_n, next_tau, next_factors)
            
            if time.time() > deadline - 10:
                return

    # Start DFS
    # Max exponent for 2 is ~59
    dfs(0, 60, 1, 1, [])
    
    if not candidates:
        # Fallback: product of first 15 primes
        n = 1
        for p in primes[:15]:
            n *= p
        return n

    # Sort by tau descending (higher tau usually correlates with higher tau_perp potential)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score = -1
    best_n = candidates[0][1]
    
    # Evaluate top candidates
    # We have ~100 seconds left for evaluation.
    eval_deadline = start_time + _DEADLINE - 2.0
    
    for tau, n, factors in candidates:
        if time.time() > eval_deadline:
            break
            
        # Use factors to compute tau_perp faster if we modify _tau_perp to accept them?
        # For now, call _tau_perp(n). It refactors, which is okay for small primes.
        score = _tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
        # Optional: Early stopping?
        # If we hit a very high score, maybe stop? 
        # Hard to know max. Keep going until time runs out.
        
    return best_n
