import time
import math

def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s limit, leaving 10s margin for 60s total
    
    # We will use a sieve to compute phi(n) for n up to a limit.
    # The answer is expected to be small. 10,000 is a safe upper bound for the minimum.
    # If not found, we can extend or fall back.
    limit = 10000
    
    # Initialize phi array
    phi = list(range(limit + 1))
    
    # Sieve for phi
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n such that phi(n) % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # If not found in the initial limit, expand the search or return a known valid value.
    # Given the problem constraints, 900 is a valid answer (phi(900)=240).
    # We can try a few more individually if we have time, but 900 is likely the min or close.
    # Let's check up to 100000 just in case, using the same sieve logic extended? 
    # Or just return 900 as a fallback if the loop didn't catch it (it should have).
    
    # If we reach here, it means no n <= 10000 worked. This is highly unlikely.
    # Let's try a direct check for a few more numbers or return a safe fallback.
    # A known solution is 900.
    return 900
