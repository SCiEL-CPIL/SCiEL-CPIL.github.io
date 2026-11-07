import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 256
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Initial set: 1 and all prime powers
    # We will represent the set as a dict: prime -> value
    # Primes not in the dict are "unused" (value 0)
    # Initially, all primes are used with their max power
    
    # Identify potential merges
    # A merge is a pair (p, q) such that p*q <= N and p*q > P_p + P_q
    # Gain = p*q - P_p - P_q
    merges = []
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i+1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                merges.append((gain, p, q, prod))
                
    # Sort by gain descending
    merges.sort(reverse=True)
    
    # Greedy selection of disjoint merges
    # Since the graph is sparse and gains are large, greedy is likely optimal
    # Or use simple backtracking if needed, but greedy is fast
    used_primes = set()
    selected_merges = []
    
    for gain, p, q, prod in merges:
        if p not in used_primes and q not in used_primes:
            selected_merges.append((p, q, prod))
            used_primes.add(p)
            used_primes.add(q)
            
    # Construct final result
    result = [1]
    for p in primes:
        if p in used_primes:
            # Find the merge involving p
            # We can just add the product when we see the first prime
            pass
        else:
            result.append(prime_powers[p])
            
    # Add the merged products
    for p, q, prod in selected_merges:
        result.append(prod)
        
    # Verify
    # Check distinct
    if len(set(result)) != len(result):
        # Fallback
        return [1] + [prime_powers[p] for p in primes]
        
    # Check range
    for x in result:
        if x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]
            
    # Check coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]
                
    return result
