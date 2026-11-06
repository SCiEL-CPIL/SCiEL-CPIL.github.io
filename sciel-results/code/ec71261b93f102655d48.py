import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55
    N = 1000000
    
    # Precompute smallest prime factor (SPF) for all numbers up to N
    # spf[i] will hold the smallest prime factor of i
    spf = list(range(N + 1))
    
    # Sieve to find smallest prime factors
    # We only need to iterate up to sqrt(N)
    limit = int(N**0.5)
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples of i
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to compute phi(n) using the SPF table
    def compute_phi(n):
        if n == 1:
            return 1
        phi = n
        x = n
        while x > 1:
            p = spf[x]
            phi -= phi // p
            while x % p == 0:
                x //= p
        return phi

    # Find the smallest n such that phi(n) is divisible by 120
    for n in range(1, N + 1):
        if time.time() > deadline:
            # Fallback to a known valid answer
            # phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
            return 143
        if compute_phi(n) % 120 == 0:
            return n
            
    # Should not reach here
    return 143
# EVOLVE-BLOCK-END
