import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 80.0 # Safety margin before 90s limit
    
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_val = float('inf')
    best_a = 1
    
    # We can optimize by only checking a up to N//2
    # due to symmetry max(P(a), P(b)) is same for (a, b) and (b, a)
    for a in range(1, N // 2 + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            # If we find the theoretical minimum for this parity, we can stop.
            # N is odd, so one is even, one is odd.
            # Min prime for odd > 1 is 3.
            # If max is 3, both must be 3-smooth.
            # This is the absolute minimum possible for N>2 odd.
            if best_val == 3:
                break
                
    return [best_a, N - best_a]
