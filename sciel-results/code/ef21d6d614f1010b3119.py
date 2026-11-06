import time

def construct():
    N = 123457
    start_time = time.time()
    
    # 1. Precompute largest prime factor (lpf) and list of primes
    lpf = [1] * (N + 1)
    primes = []
    
    for i in range(2, N + 1):
        if lpf[i] == 1:
            primes.append(i)
            # Mark multiples
            # Optimization: inner loop can be optimized but N is small
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Binary search on the minimal B (which must be a prime)
    # We search over the list of primes
    low = 0
    high = len(primes) - 1
    best_pair = None
    
    # Safety margin check
    def time_ok():
        return time.time() - start_time < 50 # 50s margin for 60s limit

    while low <= high and time_ok():
        mid = (low + high) // 2
        B = primes[mid]
        
        found_pair = None
        # Search for a in [1, N//2]
        # If a > N//2, then b < N//2, and we would have found it when a was b.
        # We just need one valid pair.
        for a in range(1, N // 2 + 1):
            if lpf[a] <= B and lpf[N - a] <= B:
                found_pair = (a, N - a)
                break
        
        if found_pair:
            best_pair = found_pair
            high = mid - 1 # Try to find a smaller B
        else:
            low = mid + 1   # Need a larger B

    # If binary search didn't find a pair (should not happen), fallback
    if best_pair is None:
        # Fallback to a known valid pair, e.g., [1, N]
        # P(1)=1, P(N) is some prime factor.
        # Or just return the seed result if time ran out
        if time_ok():
             # Try to find any pair with a larger B, e.g., N
             # But primes list goes up to N, so high would have covered it.
             # If we exited due to time, return best known or seed
             return [1, 123456]
        else:
             return [1, 123456]

    return list(best_pair)
