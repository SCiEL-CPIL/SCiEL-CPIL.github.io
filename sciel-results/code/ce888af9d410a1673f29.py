"""Combined program using a Sieve of Eratosthenes approach to compute phi(n)."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi for all numbers up to 1,000,000, then scans for the minimum.
    """
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # The answer is likely small, but we sieve the full range to be structurally robust.
    # 1,000,000 integers fit easily in memory.
    N = 1000000
    
    # Initialize phi array. phi[i] will hold the value of phi(i).
    # Initially, phi[i] = i.
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes for phi
    # For each prime p, update phi[j] for all multiples j of p.
    # phi[j] *= (1 - 1/p) => phi[j] -= phi[j] // p
    for i in range(2, N + 1):
        # Check time to be safe, though sieve is fast
        if time.time() > deadline:
            break
            
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
        
        # Optional: early exit if we've passed a known bound? 
        # No, we need the minimum, so we must compute phi for small n first.
        # The sieve computes all values. We scan afterwards.
        
    # Scan for the smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (shouldn't reach here)
    return N
# EVOLVE-BLOCK-END
