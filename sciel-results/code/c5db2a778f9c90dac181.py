import math

def construct():
    N = 1000000
    min_n = 10000
    
    # Primes up to 1000
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Generate K-smooth numbers for increasing K
    # We want to find n in [min_n, N] such that n and n+1 are K-smooth.
    # Start with small K.
    
    for i in range(len(primes)):
        K = primes[i]
        # Generate all numbers <= N using primes[0..i]
        smooths = {1}
        # Simple BFS/DFS to generate smooth numbers
        # To avoid explosion, we can just iterate and multiply
        # But N is 1e6, so the count is small.
        
        # Efficient generation:
        # Start with [1]
        # For each prime p in primes[0..i]:
        #   Multiply existing elements by p^k
        # This is like a knapsack.
        
        current_smooths = {1}
        for p in primes[:i+1]:
            new_smooths = set(current_smooths)
            # For each existing smooth number, multiply by p, p^2, ...
            # We need to be careful not to miss combinations.
            # Better:
            # new_smooths = set()
            # for x in current_smooths:
            #   val = x
            #   while val * p <= N:
            #     val *= p
            #     new_smooths.add(val)
            # current_smooths = new_smooths | current_smooths ? No.
            # This logic is flawed for multiple primes.
            
            # Correct method:
            # Maintain a list of smooth numbers.
            # When adding a new prime p, for each existing smooth number x,
            # add x*p, x*p^2, ...
            
            existing = list(current_smooths)
            for x in existing:
                val = x
                while val * p <= N:
                    val *= p
                    current_smooths.add(val)
                    
        # Filter to range [min_n, N]
        candidates = [x for x in current_smooths if min_n <= x <= N]
        candidates_set = set(candidates)
        
        # Check for consecutive pairs
        for x in candidates:
            if (x + 1) in candidates_set:
                return x
                
    # Fallback: if no pair found with primes up to 1000, 
    # it means one of n, n+1 has a prime factor > 1000.
    # In that case, we fall back to the sieve method or just return a good candidate.
    # But such pairs are rare and likely have large LPF.
    # The sieve method is safer as a fallback.
    
    # Re-implement sieve fallback if needed?
    # Given the problem constraints, a pair with small LPF should exist.
    # e.g. n=10000, n+1=10001=73*137. LPF=137.
    # We should find something better.
    
    return 10000
