"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The task requires finding the smallest integer n in [1, 1000000]
    # such that phi(n) is divisible by 120.
    
    # 120 = 2^3 * 3 * 5.
    # We can precompute the answer or find a small candidate quickly.
    # Let's check small numbers manually or via a fast sieve.
    
    # For efficiency, we can just return a known small candidate if we verify it.
    # However, to be robust and "minimize n", we should find the true minimum.
    
    # Let's use a sieve to compute phi(n) for n up to a reasonable limit, say 10000.
    # If the minimum is larger, we can expand, but it's likely small.
    
    # Actually, let's just compute phi for n from 1 upwards until we find one.
    # Since the search space is small for the minimum, a simple loop with a fast phi
    # or a precomputed list is best.
    
    # Let's use a sieve up to 10000 first.
    N = 10000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # If not found in [1, 10000], we can extend the search or return a known valid number.
    # The seed program returned 14400.
    # Let's check if 14400 is the minimum or if there's a smaller one between 10000 and 14400.
    # Given the constraints and typical distributions, the minimum is likely < 10000.
    # For safety, if we didn't find it in 1..10000, we can do a quick check up to 20000.
    
    N2 = 20000
    if N2 > N:
        # Extend the sieve
        # This is a bit complex to extend a sieve easily.
        # Let's just return a known valid number if not found in 1..10000.
        # But wait, we can just do a simple loop with a phi function for the remaining range.
        
        def _phi(n):
            result = n
            p = 2
            m = n
            while p * p <= m:
                if m % p == 0:
                    while m % p == 0:
                        m //= p
                    result -= result // p
                p += 1
            if m > 1:
                result -= result // m
            return result

        for n in range(N + 1, 1000001):
            if _phi(n) % 120 == 0:
                return n
    
    # Fallback (should not be reached if the minimum is small)
    return 14400

# EVOLVE-BLOCK-END
