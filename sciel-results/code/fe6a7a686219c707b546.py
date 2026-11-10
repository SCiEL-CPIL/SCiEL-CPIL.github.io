import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    Uses a linear sieve for smallest prime factors to compute phi(n) in O(N).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Upper bound is 1,000,000 per task contract. 
    # We can likely find the answer much earlier, but we must ensure we 
    # don't miss it if it's near the boundary. 
    # However, a linear sieve up to 1,000,000 is very fast (< 1s in Python).
    limit = 1000000
    
    # Smallest Prime Factor (SPF) sieve
    # spf[i] will store the smallest prime factor of i
    spf = list(range(limit + 1))
    
    # Standard SPF sieve
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute phi(n) for all n up to limit
    # phi[1] = 1
    phi = [0] * (limit + 1)
    phi[1] = 1
    
    # We can compute phi using the property:
    # phi(n) = n * product(1 - 1/p) for all distinct prime factors p of n
    # Alternatively, using SPF:
    # If spf[n] == n, phi[n] = n - 1
    # Else if spf[n] == spf[n // spf[n]], phi[n] = phi[n // spf[n]]
    # Else phi[n] = phi[n // spf[n]] * (spf[n] - 1)
    
    for n in range(2, limit + 1):
        p = spf[n]
        if p == n:
            phi[n] = n - 1
        else:
            m = n // p
            if m % p == 0:
                phi[n] = phi[m]
            else:
                phi[n] = phi[m] * (p - 1)
        
        # Check condition: phi(n) divisible by 120
        if phi[n] % 120 == 0:
            # We are iterating from 2 upwards. 
            # Note: phi(1)=1, not div by 120.
            # The first n found will be the minimum.
            # We can return immediately since we scan in increasing order.
            # This avoids computing the rest of the array if the answer is small.
            return n

    # Fallback if no such n found in [1, limit] (should not happen for limit=1e6)
    return limit
# EVOLVE-BLOCK-END
