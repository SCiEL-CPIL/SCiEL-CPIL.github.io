"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need to find the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    # Theoretical minimum is 143 (phi(143)=120).
    # We use a sieve to compute phi(n) for all n up to a reasonable limit, 
    # or just check up to 143 since we know it's the min.
    # However, to be safe and general, we compute up to a small limit 
    # that is guaranteed to contain the solution, or use the full sieve 
    # if performance allows (it does).
    
    # Since we know 143 is the smallest, we can just return it.
    # But to be "coherent" with the task of finding it, we can implement the sieve.
    # The sieve for N=1000000 is very fast (<0.1s).
    
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
