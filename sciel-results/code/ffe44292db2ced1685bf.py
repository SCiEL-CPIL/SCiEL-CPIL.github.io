import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = 90.0
    
    # Use a linear sieve to compute phi(n) for n up to 1,000,000 in O(N) time.
    # This is faster than the O(N log log N) sieve used previously.
    limit = 1000000
    phi = [0] * (limit + 1)
    phi[1] = 1
    is_prime = [True] * (limit + 1)
    primes = []
    
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            if i * p > limit:
                break
            is_prime[i * p] = False
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
                
    # Find the smallest n with phi[n] % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            # Check time just in case
            if time.time() > deadline - 1.0:
                return n
            return n
            
    # Fallback if not found (should not happen)
    return 143
