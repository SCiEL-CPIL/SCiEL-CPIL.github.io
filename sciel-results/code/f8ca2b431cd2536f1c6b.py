import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe
    
    N = 1000001
    # Linear Sieve to find Smallest Prime Factor (SPF)
    spf = [0] * (N + 1)
    spf[1] = 1
    primes = []
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
    
    # Check time
    if time.time() > deadline:
        return 10000

    # Precompute Largest Prime Factor (LPF)
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        p = spf[i]
        # lpf[i] is max(p, lpf[i//p])
        if i // p == 1:
            lpf[i] = p
        else:
            lpf[i] = max(p, lpf[i // p])
            
    # Check time
    if time.time() > deadline:
        return 10000

    # Search for n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])
    
    # Local variables for speed
    lpf_local = lpf
    for n in range(10001, 1000001):
        val = lpf_local[n]
        val_next = lpf_local[n + 1]
        if val > val_next:
            current_max = val
        else:
            current_max = val_next
            
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
        # Early exit if we find a very small value, e.g., 3 or 2?
        # Since n >= 10000, n and n+1 are composite or prime.
        # If n is a power of 2 and n+1 is a power of 3? Unlikely to get very small LPF.
        # The minimum possible LPF for numbers > 10000 is likely around 7-13.
        # If we find best_val == 3, we can stop. But 3^k <= 1000000 -> k<=12.
        # Powers of 2: 2^13=8192, 2^14=16384.
        # If n=16383 (3*5*1093?), no.
        # Let's just run the full loop, it's fast.
        
        if time.time() > deadline:
            break

    return best_n
