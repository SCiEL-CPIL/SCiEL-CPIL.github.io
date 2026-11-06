"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    LIMIT = 1000000
    
    # We need to find the smallest n such that 120 | phi(n).
    # Let's use a sieve approach to compute phi(n) for all n up to LIMIT.
    # However, we can stop early as soon as we find the smallest n.
    
    # Initialize phi array with phi[i] = i
    phi = list(range(LIMIT + 1))
    
    # Sieve of Eratosthenes style for phi
    # For each prime p, update phi for multiples of p
    for i in range(2, LIMIT + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, LIMIT + 1, i):
                phi[j] -= phi[j] // i
        
        # Check if phi[i] is divisible by 120
        if phi[i] % 120 == 0:
            # Since we iterate i in increasing order, the first such i is the answer
            return i
        
        # Time check
        if time.time() > deadline:
            # If we've run out of time, return the current best guess
            # But since we're iterating in order, we haven't found one yet.
            # This shouldn't happen for LIMIT=1e6 as the loop is fast.
            # Fallback: return a known valid number, e.g., 120 (if phi(120)%120==0?)
            # phi(120) = 120 * (1-1/2) * (1-1/3) * (1-1/5) = 120 * 1/2 * 2/3 * 4/5 = 32. 32%120 != 0.
            # Let's just return 1 as a fallback (invalid but safe from crash) or a precomputed small one.
            # Actually, let's precompute a small candidate just in case.
            # phi(240) = 240 * 1/2 * 2/3 * 4/5 = 64. No.
            # phi(360) = 360 * 1/2 * 2/3 * 4/5 = 96. No.
            # phi(720) = 720 * 1/2 * 2/3 * 4/5 = 192. No.
            # phi(1200) = 1200 * 1/2 * 2/3 * 4/5 = 320. 320 % 120 = 80.
            # Let's find one quickly:
            # We need phi(n) to be a multiple of 120.
            # Try n=240? No.
            # Try n= 2^4 * 3 * 5 = 16*15 = 240. phi=64.
            # Try n= 2^3 * 3 * 5 * 7 = 8*105 = 840.
            # phi(840) = 840 * 1/2 * 2/3 * 4/5 * 6/7 = 840 * (1/2 * 2/3 * 4/5 * 6/7)
            # = 840 * (1/7 * 4/5 * 6/7) ... wait.
            # 1/2 * 2/3 = 1/3.
            # 1/3 * 4/5 = 4/15.
            # 4/15 * 6/7 = 24/105 = 8/35.
            # phi(840) = 840 * 8/35 = 24 * 8 = 192. No.
            
            # Let's just return 120*120 as in the seed, which is 14400.
            # phi(14400) = phi(144 * 100) = phi(16*9 * 4*25) = phi(2^6 * 3^2 * 5^2)
            # = 14400 * (1/2) * (2/3) * (4/5) = 14400 * 8/30 = 14400 * 4/15 = 960 * 4 = 3840.
            # 3840 / 120 = 32. Yes, 14400 is a valid answer.
            return 14400

    # If the loop finishes without finding one (unlikely for LIMIT=1e6), return 14400
    return 14400
# EVOLVE-BLOCK-END
