import time

def construct():
    """
    Finds integer n in [10000, 1000000] minimizing the largest prime factor of n(n+1).
    Uses a sieve to precompute largest prime factors for O(1) lookup.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    N = 1000000
    MIN_N = 10000
    
    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # Iterate through potential primes
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Check time after sieve
    if time.time() > deadline:
        # Fallback: return a known good value or the minimum n
        # 10000 is valid but not optimal. 
        # We can't easily compute the best without the sieve.
        # But the sieve should finish in <1s.
        return 10000

    best_n = MIN_N
    min_lpf_max = float('inf')
    
    # Search for n in [10000, 999999] such that n+1 <= 1000000
    # We want to minimize max(lpf[n], lpf[n+1])
    for n in range(MIN_N, N):
        # Calculate largest prime factor of the product n(n+1)
        # This is max(lpf[n], lpf[n+1])
        curr_max = lpf[n]
        if lpf[n+1] > curr_max:
            curr_max = lpf[n+1]
            
        if curr_max < min_lpf_max:
            min_lpf_max = curr_max
            best_n = n
            
        # Optional: Early exit if we find 2? 
        # Impossible for n >= 10000.
        # If we find a very small number, we can continue to ensure we found the minimum.
        
        # Time check inside loop just in case, though unlikely needed
        if (n - MIN_N) % 100000 == 0 and time.time() > deadline:
            break

    return best_n
