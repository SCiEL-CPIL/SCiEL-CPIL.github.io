"""Find smallest n in [1,1000000] with phi(n) divisible by 120."""
import time

def construct():
    """Return smallest n in [1,1000000] such that phi(n) % 120 == 0."""
    start = time.time()
    deadline = start + 85  # 5s safety margin before 90s kill
    
    # Use a sieve to compute phi for all n up to 1000000
    MAX_N = 1000000
    phi = list(range(MAX_N + 1))
    
    for i in range(2, MAX_N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, MAX_N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi[n] % 120 == 0
    for n in range(1, MAX_N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not reach here)
    return 120 * 120
