import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    # Precompute primes up to the maximum gap size
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes = sieve(max_gap)
    
    # For each gap size, precompute the list of primes strictly less than that gap
    # This allows O(1) access to the relevant primes for each gap
    prime_checkers = {}
    for g in range(2, max_gap + 1):
        prime_checkers[g] = [p for p in primes if p < g]
        
    results = []
    
    for lo, hi in P['gaps']:
        gap = hi - lo
        relevant_primes = prime_checkers.get(gap, [])
        
        witness = None
        # Check each integer strictly between lo and hi
        for m in range(lo + 1, hi):
            is_valid = True
            for p in relevant_primes:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                witness = m
                break
        
        # Convert to int to ensure type safety, or None
        if witness is not None:
            results.append(int(witness))
        else:
            results.append(None)
            
    return results
