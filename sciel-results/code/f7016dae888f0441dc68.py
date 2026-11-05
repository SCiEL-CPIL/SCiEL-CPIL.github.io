import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 148.0  # 2s margin for safety
    
    # Generate first 30 primes
    primes = []
    candidate = 2
    while len(primes) < 30:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    
    # For n <= 10^18, the maximum number of distinct prime factors is 15 (2*3*5*...*47 ~ 6e17)
    # The maximum exponent for 2 is 59 (2^59 ~ 5.7e17)
    
    best_n = 2
    best_score = 0
    
    def calc_score_fast(n):
        if n < 2:
            return 0
        
        # Factorize n
        factors = []
        temp_n = n
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                exp = 0
                while temp_n % p == 0:
                    temp_n //= p
                    exp += 1
                factors.append((p, exp))
        if temp_n > 1:
            factors.append((temp_n, 1))
        
        # Generate divisors using NumPy for speed if possible, or optimized list
        # Since n <= 10^18, the number of divisors is at most ~100k.
        # Standard list generation is fast enough in Python if done efficiently.
        
        divisors = np.array([1], dtype=np.int64)
        for p, e in factors:
            # Create powers of p
            powers = np.array([p**i for i in range(e+1)], dtype=np.int64)
            # Outer product
            divisors = np.unique(np.multiply.outer(divisors, powers).flatten())
            
        # Sort
        divisors = np.sort(divisors)
        
        # Calculate score
        # gcd of adjacent elements
        # We can use a loop, but np.gcd works on arrays
        # However, we need to check pairs (d[i], d[i+1])
        # This is a vectorized operation on shifted arrays
        
        if len(divisors) < 2:
            return 0
        
        d1 = divisors[:-1]
        d2 = divisors[1:]
        
        # np.gcd returns an array
        gcds = np.gcd(d1, d2)
        
        # Count where gcd == 1
        score = np.sum(gcds == 1)
        
        return int(score)

    # Recursive search for best exponent vector
    # We assume exponents are non-increasing: e1 >= e2 >= ...
    # This is a standard property for maximizing divisor count or similar structural metrics
    # while keeping n small. For "coprime adjacent divisors", we want many distinct prime
    # factors and a specific structure.
    
    # Actually, for maximizing coprime adjacent divisors, we want the divisors to be
    # "spread out" in terms of prime factors.
    # A known result for similar problems is that primorials (product of first k primes)
    # or products of first k primes with small exponents are good candidates.
    
    # Let's do a DFS on exponents.
    # Max depth: 15 (since product of first 15 primes is < 10^18)
    
    def dfs(index, max_exp, current_n, current_exps):
        nonlocal best_n, best_score
        
        if time.time() > deadline:
            return
        
        if index == len(primes):
            if current_n > 1:
                s = calc_score_fast(current_n)
                if s > best_score:
                    best_score = s
                    best_n = current_n
            return
        
        p = primes[index]
        
        # If current_n * p > 10^18, we can't include this prime or any larger one
        if current_n * p > 10**18:
            # We can stop exploring deeper if we can't even multiply by the next prime
            # But we might have already added this prime in previous steps? No, index is current.
            # If we can't multiply by p, we must stop adding primes.
            # However, we might have skipped this prime? No, the logic adds primes sequentially.
            # If we don't add p, we should still check the current state?
            # The "current state" is defined by the exponents chosen so far.
            # If we stop adding primes, we evaluate the current candidate.
            if current_n > 1:
                s = calc_score_fast(current_n)
                if s > best_score:
                    best_score = s
                    best_n = current_n
            return
        
        # Try exponents from max_exp down to 0
        # Optimization: if max_exp is large, limit it.
        # 2^59 is max.
        
        # Pruning: if current_n * (p ** max_exp) > 10^18, reduce max_exp
        while max_exp > 0 and current_n * (p ** max_exp) > 10**18:
            max_exp -= 1
            
        for e in range(max_exp, -1, -1):
            new_n = current_n * (p ** e)
            if new_n > 10**18:
                continue
                
            # Recurse
            # Next prime must have exponent <= e
            dfs(index + 1, e, new_n, current_exps + [e])
            
            # Early termination?
            # If time is up, return
            if time.time() > deadline:
                return
            
            # If e == 0, we are done adding primes for this branch
            # The recursion will handle the "end" case at index == len(primes)
            # But we can also evaluate here if we decide to stop?
            # The recursion goes to index+1. If we don't add p (e=0), we still go to index+1.
            # This means we will try to add p_{index+1} with max_exp=0, which means it won't be added.
            # This effectively explores all subsets of primes with non-increasing exponents.
            
    # Start DFS
    # Initial max_exp for 2 is 60
    dfs(0, 60, 1, [])
    
    return best_n

# EVOLVE-BLOCK-END
