"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 40  # Leave 20s margin for safety
    
    # We need to find the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # The seed program returns 120 * 120 = 14400.
    # Let's check if there are smaller numbers.
    # phi(n) is divisible by 120 = 2^3 * 3 * 5.
    
    # Strategy: Compute phi(n) for n from 1 to N (where N is some limit, say 20000 or so)
    # and check the condition. Since we want to minimize n, we can just iterate n=1,2,3...
    # and stop at the first one that satisfies the condition.
    
    # However, computing phi(n) individually for each n is O(sqrt(n)) which might be slow
    # if the answer is large. But the answer is likely small.
    # Let's use a sieve to compute phi(n) for all n up to a limit.
    
    # Let's estimate the limit. The seed is 14400.
    # Let's compute phi for n up to 20000 using a sieve.
    # This is very fast.
    
    N = 20000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                if time.time() > deadline:
                    return 14400  # Fallback to seed if we run out of time (unlikely)
                phi[j] = phi[j] // i * (i - 1)
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # If not found in [1, 20000], return the seed value which is known to be valid.
    # phi(14400) = 14400 * (1-1/2) * (1-1/3) * (1-1/5) = 14400 * 1/2 * 2/3 * 4/5 = 14400 * 4/15 = 3840.
    # 3840 / 120 = 32. So 14400 is valid.
    return 14400
# EVOLVE-BLOCK-END
