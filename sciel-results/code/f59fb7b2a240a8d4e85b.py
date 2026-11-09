# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # We need phi(n) % 120 == 0
    # 120 = 8 * 3 * 5 = 2^3 * 3 * 5
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    # For phi(n) to be divisible by 120, it must be divisible by 8, 3, and 5.
    
    # Let's just compute phi for n from 1 upwards and check.
    # Since we want to minimize n, we can stop as soon as we find one.
    # However, computing phi from scratch for each n is O(sqrt(n)).
    # A sieve approach is more efficient for checking many numbers, but since we stop early,
    # we might not need a full sieve up to 1000000.
    # But wait, if the answer is small, direct computation is fine.
    # If the answer is large, a sieve up to that limit is better.
    
    # Let's estimate: how large is the smallest n?
    # phi(n) >= 120.
    # For n=1, phi=1
    # ...
    # Let's just try a sieve up to a reasonable limit first, say 10000.
    # If not found, expand.
    
    # Actually, let's just do a simple loop with an efficient phi calculator.
    # Or better, use a sieve to precompute phi up to some limit.
    # If the limit is 1000000, the sieve is O(N log log N) which is very fast in Python if optimized, 
    # or using NumPy.
    
    # Let's use NumPy for speed if available, otherwise standard library.
    # The task says NumPy is available.
    
    import numpy as np
    
    # Let's try to find the answer quickly.
    # We can check n=1, 2, ...
    # But doing phi(n) individually is slow.
    # Sieve is better.
    
    # Let's set a limit. If we don't find it in 10000, we go to 100000, then 1000000.
    # But actually, we can just run a sieve up to 1000000 directly. It's fast enough.
    
    # Initialize phi array
    limit = 1000000
    phi = np.arange(1, limit + 1)
    
    # Sieve of Eratosthenes style for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] = phi[j] // i * (i - 1)
    
    # Now find the smallest n such that phi[n-1] % 120 == 0 (since phi is 1-indexed in our array logic? 
    # No, phi[i] corresponds to phi(i) if we initialized correctly?
    # Let's trace:
    # phi = [1, 2, 3, ..., N]
    # For prime p, phi[p] = p.
    # Update: for j multiple of p: phi[j] -= phi[j] // p.
    # This is the standard sieve.
    # phi[i] holds phi(i).
    
    # Find smallest n >= 1
    # We need phi(n) % 120 == 0
    # Check indices 0 to limit-1 corresponding to n=1 to limit
    # phi[0] is phi(1) = 1
    # phi[1] is phi(2) = 1
    # ...
    
    # Vectorized check
    valid = (phi % 120 == 0)
    
    # Find first True
    indices = np.where(valid)[0]
    
    if len(indices) > 0:
        n = indices[0] + 1  # Because phi[0] is for n=1
        return n
    else:
        # Fallback if something went wrong (shouldn't happen within 1000000)
        return 1

# EVOLVE-BLOCK-END
