# EVOLVE-BLOCK-START
import time

def construct():
    # We need to find the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # We can compute phi(n) for all n up to 1000000 using a sieve.
    
    N = 1000000
    phi = list(range(N + 1))
    
    # Sieve for Euler's totient function
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not be reached)
    return 14400
# EVOLVE-BLOCK-END
